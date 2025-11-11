# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from pkg_resources import packaging

import dataclasses
import fire
import random
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR, SequentialLR
from transformers import (
    # LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
)
from transformers import AutoTokenizer
from llama_recipes.custom_models.modeling_llama_3_2 import LlamaForCausalLM
from llama_recipes.custom_models.modules.shape_tokenizer import get_shape_tokenizer #ShapeTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_shape2code_config as TRAIN_CONFIG
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.fsdp_utils import hsdp_device_mesh
from llama_recipes.utils.train_utils import (
    # train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from llama_recipes.utils.train_utils_shape2code import train_shape2code
from accelerate.utils import is_xpu_available

from llama_recipes.utils.miscellaneous import print_message, partial_load
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from datetime import timedelta
import time

import yaml
import shutil
import pdb

def setup_wandb(train_config, fsdp_config, **kwargs):
    try: 
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama_recipes.configs import wandb_config as WANDB_CONFIG
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    wandb.login(key=train_config.wandb_key)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)
    return run

def multi_process_load_dataset(tokenizer, dataset_config, split, enable_fsdp, rank):
    if not enable_fsdp or rank == 0:
        print('rank 0 processing data')
        dataset = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split=split,
        )
        print(f"--> Training Set Length = {len(dataset)}")
    else:
        print('rank %s waiting rank 0 processing data' % str(rank))

    if enable_fsdp:
        dist.barrier()
        # rank 0 first time processing data may take more time
    
    if enable_fsdp and rank > 0:
        print('rank %s reading rank 0 cached data' % str(rank))
        dataset = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split=split,
        )
        print(f"--> Training Set Length = {len(dataset)}")

    if enable_fsdp:
        dist.barrier()

    return dataset

def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    wandb_run = None

    if train_config.use_wandb:
        if not train_config.enable_fsdp or rank==0:
            try:
                wandb_run = setup_wandb(train_config, fsdp_config, **kwargs)
            except:
                print('wandb init failed, continue training without wandb')

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            # device_map='cuda:%d' % local_rank,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        )

    # Load the tokenizer and add special tokens
    # tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name
    )
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.use_peft:
        time.sleep(10*local_rank)
        peft_config = generate_peft_config(train_config, kwargs)
        if not train_config.resume_training_dir is None:
            try:
                model = PeftModel.from_pretrained(model, train_config.resume_training_dir, is_trainable=True)
                print_message(train_config.enable_fsdp, rank if train_config.enable_fsdp else None, 
                        'peft model ckpt has been loaded')
            except Exception as error:
                print_message(train_config.enable_fsdp, rank if train_config.enable_fsdp else None, 
                        'an error occurred when loading the pretrained peft model, the error is:\n%s' % error)
                print_message(train_config.enable_fsdp, rank if train_config.enable_fsdp else None, 
                        'we initialize the peft model from scratch')
                model = get_peft_model(model, peft_config)
        else:
            model = get_peft_model(model, peft_config)
            print_message(train_config.enable_fsdp, rank if train_config.enable_fsdp else None, 
                    'peft model will be trained from scratch')
        model.print_trainable_parameters()
        if wandb_run:
            wandb_run.config.update(peft_config)

        
    hsdp_device_mesh = None
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh = hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size, sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")

    # build the shape tokenizer model
    with open(train_config.shape_tokenizer_config, 'r') as yaml_file:
        shape_tokenizer_config = yaml.safe_load(yaml_file)
    if not train_config.shape_tokenizer_init_token is None:
        shape_tokenizer_config['querier_args']['zero_init_output'] = True
        init_token_idx = tokenizer.encode(train_config.shape_tokenizer_init_token)
        assert len(init_token_idx)==2 # the first element in init_token_idx is 1, start of sentence
        init_token_idx = torch.Tensor([init_token_idx[1]]).long()
        offset_tensor = model.get_input_embeddings()(init_token_idx) # 1, embed_dim
        shape_tokenizer_config['offset_tensor'] = offset_tensor[0] # embed_dim
    # shape_tokenizer_model = ShapeTokenizer(**shape_tokenizer_config)
    shape_tokenizer_model = get_shape_tokenizer(shape_tokenizer_config)

    # load shape_tokenizer_ckpt
    start_epoch = 0
    if not train_config.resume_training_dir is None:
        time.sleep(10*local_rank)
        shape_tokenizer_ckpt = torch.load(os.path.join(train_config.resume_training_dir, 'shape_tokenizer.pt'), 
                                map_location='cpu', weights_only=False)
        # shape_tokenizer_model.load_state_dict(shape_tokenizer_ckpt['model_state_dict'])
        shape_tokenizer_model = partial_load(shape_tokenizer_model, shape_tokenizer_ckpt['model_state_dict'], allow_shape_mismatch=True)
        print_message(train_config.enable_fsdp, rank if train_config.enable_fsdp else None, 
                    'shape tokenizer model ckpt has been loaded')
        if train_config.resume_epoch_count:
            start_epoch = shape_tokenizer_ckpt['epoch']
            print_message(train_config.enable_fsdp, rank if train_config.enable_fsdp else None, 
                    'resume training from epoch %d' % start_epoch)
    
    # move the model to cuda and setup ddp
    if train_config.enable_fsdp:
        shape_tokenizer_model.to("cuda")
        # print('move tokenizer model to ddp rank %d local_rank %d world_size %d' % (rank, local_rank, world_size))
        shape_tokenizer_model = DistributedDataParallel(shape_tokenizer_model, device_ids=[local_rank], find_unused_parameters=True)
        # shape_tokenizer_model = DistributedDataParallel(shape_tokenizer_model, device_ids=[local_rank], find_unused_parameters=False)
        # print('isinstance(shape_tokenizer_model, DistributedDataParallel)', isinstance(shape_tokenizer_model, DistributedDataParallel))
    else:
        shape_tokenizer_model.to("cuda")
    
    # save all configs to output_dir
    if not train_config.enable_fsdp or rank==0:
        os.makedirs(train_config.output_dir, exist_ok=True)
        shutil.copyfile(train_config.shape_tokenizer_config, 
            os.path.join(train_config.output_dir, 'config.yaml'))
        
        with open(os.path.join(train_config.output_dir, 'config_full_training.yml'), 'w') as yaml_file:
            yaml.dump(train_config.__dict__, yaml_file, default_flow_style=False, sort_keys=False)
        
        with open(os.path.join(train_config.output_dir, 'config_command_line_args.yml'), 'w') as yaml_file:
            yaml.dump(kwargs, yaml_file, default_flow_style=False, sort_keys=False)
        
        if train_config.enable_fsdp:
            with open(os.path.join(train_config.output_dir, 'fsdp_config.yml'), 'w') as yaml_file:
                yaml.dump(fsdp_config.__dict__, yaml_file, default_flow_style=False, sort_keys=False)

        if train_config.use_peft:
            with open(os.path.join(train_config.output_dir, 'peft_config.yml'), 'w') as yaml_file:
                yaml.dump(peft_config.__dict__, yaml_file, default_flow_style=False, sort_keys=False)
        
    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
        
        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()
        time.sleep(10*local_rank)
        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")

    # kwargs are command line provided args
    dataset_config = generate_dataset_config(train_config, kwargs)
    if not train_config.enable_fsdp or rank==0:
        with open(os.path.join(train_config.output_dir, 'config_dataset.yml'), 'w') as yaml_file:
            yaml.dump(dataset_config.__dict__, yaml_file, default_flow_style=False, sort_keys=False)

     # Load and preprocess the dataset for training and validation
    dataset_train = multi_process_load_dataset(tokenizer, dataset_config, "train", 
                        train_config.enable_fsdp, rank)
    dataset_val = multi_process_load_dataset(tokenizer, dataset_config, "test", 
                        train_config.enable_fsdp, rank)

    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=train_config.lr_scheduler_step_size, gamma=train_config.gamma, verbose=True)
    # build optimizer for shape tokenizer
    optimizer_shape_tokenizer = optim.AdamW(
            shape_tokenizer_model.parameters(),
            lr=train_config.shape_tokenizer_lr,
            weight_decay=train_config.shape_tokenizer_weight_decay,
        )
    scheduler_shape_tokenizer = StepLR(optimizer_shape_tokenizer, 
                    step_size=train_config.shape_tokenizer_lr_scheduler_step_size, 
                    gamma=train_config.shape_tokenizer_lr_scheduler_gamma, verbose=True)
    if not train_config.resume_training_dir is None:
        if train_config.load_optimizer_state_dict:
            optimizer_shape_tokenizer.load_state_dict(shape_tokenizer_ckpt['optimizer_state_dict'])
            print_message(train_config.enable_fsdp, rank if train_config.enable_fsdp else None, 
                    'shape tokenizer optimizer ckpt has been loaded')
        if train_config.load_scheduler_state_dict:
            scheduler_shape_tokenizer.load_state_dict(shape_tokenizer_ckpt['scheduler_state_dict'])
            print_message(train_config.enable_fsdp, rank if train_config.enable_fsdp else None, 
                    'shape tokenizer scheduler ckpt has been loaded')


    # Start the training process
    results = train_shape2code(
        model,
        shape_tokenizer_model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        optimizer_shape_tokenizer,
        scheduler_shape_tokenizer,
        train_config.gradient_accumulation_steps,
        train_config,
        dataset_config, 
        fsdp_config if train_config.enable_fsdp else None,
        local_rank,
        rank,
        wandb_run,
        start_epoch=start_epoch,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
        if train_config.use_wandb:
            for k,v in results.items():
                wandb_run.summary[k] = v

if __name__ == "__main__":
    fire.Fire(main)
