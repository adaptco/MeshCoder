# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class train_shape2code_config:
    model_name: str="PATH/to/LLAMA/7B"
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=4
    batching_strategy: str="packing" #alternative: padding
    context_length: int=4096
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=3
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_wandb: bool = False # Enable wandb for experient tracking
    save_metrics: bool = False # saves training metrics to a json file for later plotting

    # new args for shape to code training
    resume_training_dir: Optional[str] = None
    # if it is not None, we will load ckpts from this dir
    load_optimizer_state_dict: bool = False # not implemented for the llama model yet
    load_scheduler_state_dict: bool = False # not implemented for the llama model yet
    resume_epoch_count: bool = False
    # these three args are only valid when resume_training_dir is not None
    # if a training is interrupted and we want to resume training, then we need to set the three args true
    # if we start a new exp and only want models to be initialized from a previous exp, we set the three args false
    

    shape_tokenizer_config: str = "src/llama_recipes/configs/shape_tokenizer_configs/config_shape_tokenizer.yml"
    shape_tokenizer_lr: float = 1e-4
    shape_tokenizer_weight_decay: float = 0.0
    shape_tokenizer_lr_scheduler_gamma: float = 0.85
    shape_tokenizer_lr_scheduler_step_size: int = 1
    shape_tokenizer_warm_up_epochs: int = 0
    batch_sampler: str = "LengthBasedBatchSampler"

    shape_tokenizer_init_token: Optional[str] = None
    # if this token is not None, we will force zero initial output of the shape tokenizer, 
    # and add an offset of the embedding of this token, possible tokens are placeholder, unknown

    # llama related additional args
    lr_scheduler_step_size: int = 1

    # maximum valid value for cd loss and normalized cd loss when computing mean cd loss during evaluations
    # those cd loss greater than max_valid_cd_loss will be excluded when computing mean to avoid extreme values affecting mean
    max_valid_cd_loss: Optional[float] = None

    wandb_key: str="test"
    visualization: bool=True # whether visualize the reconstruction results during every evaluation epoch