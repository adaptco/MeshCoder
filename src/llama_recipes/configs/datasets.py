# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class shape2code_dataset:
    dataset: str =  "shape2code_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    num_shape_tokens: int = 128
    num_points: int = 16384
    extract_numbers_and_object_type: bool = False
    number_replacement_token: Optional[str] = None
    data_dir: str = 'data/without_rotation'
    dataset_repeat: Optional[str] = None
    # we can use this arg to repeat the dataset in each data_dir, it should have the format
    # train: 10, 1, 1; validation: 1, 1, 1; test: 1, 1, 1
    num_samples: Optional[str] = None 
    # we can use this arg to subsample the data in each data_dir, it should have the format
    # train: 10000, 10000, 10000; validation: 2000, 2000, 2000; test: 2000, 2000, 2000
    # we can specify a negative value if do not want to downsample, namely, use the full dataset
    dataset_labels: Optional[str] = None
    # we can use this arg to assign a label to each dataset specified in data_dir
    # this is used to identify each dataset in the evluation phase, we can obtain the model's performance for each dataset
    # it could be: primitive, translation, rotation 
    dataset_version: str = 'single_object'
    cache_dir: str = '~/.cache/huggingface/datasets'

    augmentation_file: Optional[str] = None # e.g., src/llama_recipes/configs/dataset_configs/config_point_augmentation.yaml
    augmentation_args: Optional[dict] = None # contain args for augmentation of input point clouds, such as add holes, noises

    max_sequence_length: Optional[int] = None # max sequence length of the datasets, samples longer than max sequence length will be filtered out
    max_cd_loss: Optional[float] = None # max cd loss between pcd sampled from code generated mesh and pcd in the dataset, 
    # samples with cd loss larger than max_cd_loss will be filtered out
    pcd_path_prefix: Optional[str] = None

    num_proc: int = 1
    # number of processes to map and filter the dataset

@dataclass
class instruct2code_dataset:
    dataset: str =  "instruct2code_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"