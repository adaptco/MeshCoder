# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class train_shape_feature_predictor_config:
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
    shape_feature_predictor_config: str = "src/llama_recipes/configs/shape_tokenizer_configs/config_shape_feature_predictor.yml"
    shape_feature_predictor_lr: float = 1e-4
    shape_feature_predictor_weight_decay: float = 0.0
    shape_feature_predictor_lr_scheduler_gamma: float = 0.85
    shape_feature_predictor_lr_scheduler_step_size: int = 100
    shape_feature_predictor_warm_up_epochs: int = 0
    batch_sampler: str = "LengthBasedBatchSampler"

    shape_feature_predictor_init_token: Optional[str] = None
    # if this token is not None, we will force zero initial output of the shape tokenizer, 
    # and add an offset of the embedding of this token, possible tokens are placeholder, unknown

    # llama related additional args
    lr_scheduler_step_size: int = 1