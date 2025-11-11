
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import os
import random
import pdb

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
# import re

from llama_recipes.datasets.oss_io_utils import Npz_OSS_IO

def obtain_normalize_parameters(pcd):
    # pcd is of shape B,N,3
    minn = torch.min(pcd, dim=1, keepdim=True)[0] # B,1,3
    maxx = torch.max(pcd, dim=1, keepdim=True)[0] # B,1,3
    center = (maxx+minn) / 2 # B,1,3
    max_length = torch.max(maxx-minn, dim=2, keepdim=True)[0] # B,1,1
    return center, max_length


def count_pcd_number(dataset, num_proc=1):
    
    def add_up_number_of_pcd(sample):
        return {"pcd_sum": [sum([len(pcd_path) for pcd_path in sample])]}
    
    ds_reduce = dataset.map(add_up_number_of_pcd, input_columns="pcd_path", batched=True, 
                remove_columns=dataset.column_names, num_proc=num_proc)
    # pdb.set_trace()
    # len(ds_reduce["pcd_sum"])
    total_pcd_number = sum(ds_reduce["pcd_sum"])
    return total_pcd_number


class DatasetWrapper(Dataset):
    def __init__(self, dataset, num_points=16384, augmentation_args=None):
        self.dataset = dataset
        self.num_points = num_points
        self.augmentation_args = augmentation_args
        self.s_cluster_oss_client = None

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.dataset[int(idx)])
        
        # load point cloud
        # print(sample['pcd_path'], flush=True)
        if isinstance(sample['pcd_path'], list):
            # current_pcd_path = random.choice(sample['pcd_path'])
            pcd_idx = np.random.randint(len(sample['pcd_path']))
            current_pcd_path = sample['pcd_path'][pcd_idx]
            if 'cd_loss' in sample.keys():
                assert len(sample['cd_loss']) == len(sample['pcd_path'])
                sample['cd_loss'] = sample['cd_loss'][pcd_idx]
            sample['pcd_path'] = current_pcd_path
            
        else:
            current_pcd_path = sample['pcd_path']
        try:
            if current_pcd_path.startswith('s3://'):
                if self.s_cluster_oss_client is None:
                    self.s_cluster_oss_client = Npz_OSS_IO()
                    print('pcd path is', current_pcd_path)
                    print('we initialize the Npz_OSS_IO', flush=True)
                data = self.s_cluster_oss_client.read(current_pcd_path)
            else:
                # print(sample['pcd_path'])
                data = np.load(current_pcd_path)
        except Exception as error:
            print('error while loading pcd npz file')
            print('the error is:\n', error)
            print('the pcd path is:\n', current_pcd_path)
            data = {}
            data['points'] = (np.random.rand(50000, 3).astype(np.float32) - 0.5)*0.4 # range from -0.2 to 0.2
            data['normals'] = np.random.rand(50000, 3).astype(np.float32) - 0.5
            norm = np.sqrt((data['normals']**2).sum(axis=1, keepdims=True))
            data['normals'] = data['normals'] / (norm+1e-8)
        points = data['points'].astype(np.float32)
        normals = data['normals'].astype(np.float32)
        if len(points.shape)==3 and points.shape[0] == 1:
            points = points[0]
            normals = normals[0]
        points = np.concatenate([points, normals], axis=1)
        if not self.augmentation_args is None and 'num_points_range' in self.augmentation_args.keys():
            num_points = np.random.randint(
                self.augmentation_args['num_points_range'][0], self.augmentation_args['num_points_range'][1]+1)
            num_gt_points = self.augmentation_args['num_points_range'][1]
        else:
            num_points = self.num_points
            num_gt_points = self.num_points
        
        gt_idx = random.sample(range(points.shape[0]), num_gt_points)
        gt_idx = np.array(gt_idx)
        gt_points = points[gt_idx,:]
        sample['gt_points'] = gt_points
        
        idx = random.sample(range(points.shape[0]), num_points)
        idx = np.array(idx)
        points = points[idx,:]
        sample['points'] = points
        # sample['points'] = np.random.rand(16384, 6).astype(np.float32)
        # sample.pop('pcd_path')
        
        return sample

    def __len__(self):
        return len(self.dataset)

def string_to_dict(s):
    # s is like train: 10000, 10000, 10000; val: 2000, 2000, 2000; test: 2000, 2000, 2000
    splits = s.split(';')
    result = {}
    for split in splits:
        # split is like train: 10000, 10000, 10000
        key, values = split.split(':')
        key = key.strip()
        values = values.strip() # 10000, 10000, 10000
        values = values.split(',')
        values_list = [int(v.strip()) for v in values]
        result[key] = values_list
    return result


def get_preprocessed_shape2code(dataset_config, tokenizer, split):
    current_path = os.path.split(os.path.realpath(__file__))[0]
    # pdb.set_trace()
    # we can specify multiple data_dir sperated by comma, and we will concat them together
    data_dir_list = dataset_config.data_dir.split(',')
    data_dir_list = [data_dir.strip() for data_dir in data_dir_list]
    if not dataset_config.num_samples is None:
        num_samples = string_to_dict(dataset_config.num_samples)
    dataset_list = []

    print('loading datasets from %s' % current_path)
    dataset_labels = dataset_config.dataset_labels
    if not dataset_labels is None:
        # sometimes we observe it is already a tuple splitted by ,
        if isinstance(dataset_labels, str):
            dataset_labels = dataset_labels.split(',')
            dataset_labels = [dataset_label.strip() for dataset_label in dataset_labels]
        assert len(dataset_labels) == len(data_dir_list)

    if not dataset_config.dataset_repeat is None:
        dataset_repeat = string_to_dict(dataset_config.dataset_repeat)

    for idx, data_dir in enumerate(data_dir_list):
        print('[%d/%d] processing dataset %s' % (idx+1, len(data_dir_list), data_dir), flush=True)
        dataset = datasets.load_dataset(os.path.join(current_path, "shape2code_dataset"), split=split, 
                    data_dir=data_dir, cache_dir=dataset_config.cache_dir, trust_remote_code=True)
        if not dataset_config.num_samples is None:
            # downsample the original dataset and select the first num_samples[split][idx] samples
            # we can specify a negative value if do not want to downsample
            if num_samples[split][idx] > 0:
                dataset = dataset.select(range(num_samples[split][idx]))
        if not dataset_config.dataset_repeat is None:
            if dataset_repeat[split][idx] > 1:
                dataset = dataset.repeat(dataset_repeat[split][idx])
                print('%s set of %s dataset has been repeated %d times' % (split, data_dir, dataset_repeat[split][idx]))

        num_shape_tokens = dataset_config.num_shape_tokens
        # prompt = ("Write Blender Python script according to the following instructions:\n%s\n---\nCode:\n" % (' '.join(['placeholder']*num_shape_tokens)) )
        prompt_prefix = "Write Blender Python script according to the following instructions:\n"
        prompt_suffix = "\n---\nCode:\n"

        def apply_prompt_template(sample):
            code = sample["script"]
            if dataset_config.extract_numbers_and_object_type:
                num_start = sample["num_start"]
                num_end = sample["num_end"]
                numbers = []
                for i in range(len(num_start)-1, -1, -1):
                    numbers = [float(code[num_start[i]:num_end[i]])] + numbers
                    if not dataset_config.number_replacement_token is None:
                        code = code[0:num_start[i]] + '<NUMBER>' + code[num_end[i]:]
                # numbers = np.array(numbers).astype(np.float32)
            
            result = {
                "prompt_prefix": prompt_prefix,
                "prompt_suffix": prompt_suffix,
                "code": code,
            }
            if dataset_config.extract_numbers_and_object_type:
                result["numbers"] = numbers
            if 'cd_loss' in sample.data.keys():
                result['cd_loss'] = sample['cd_loss']
            return result
        
        dataset = dataset.map(apply_prompt_template, remove_columns=['prompt', 'script', 'id', 'num_start', 'num_end'], num_proc=dataset_config.num_proc)

        def process_pcd_path(pcd_path, current_path, data_dir, pcd_path_prefix):
            if not 'point_cloud' in pcd_path:
                # part dataset convention
                # it could record relative path or absolute path
                if pcd_path_prefix is None: # pcd npz files are at local storage
                    abs_path = os.path.join(current_path, "shape2code_dataset", data_dir, 'point_cloud', pcd_path)
                else: # pcd npz files are on oss 
                    abs_path = os.path.join(pcd_path_prefix, os.path.split(data_dir)[-1], 'point_cloud', pcd_path)
            elif 'point_cloud' in pcd_path and not pcd_path_prefix is None:
                # object dataset convention, pcd npz files are on oss 
                abs_path = os.path.join(pcd_path_prefix, os.path.split(data_dir)[-1], pcd_path)
            elif 'point_cloud' in pcd_path and pcd_path_prefix is None:
                # object dataset convention, pcd npz files are at local storage
                abs_path = os.path.join(current_path, "shape2code_dataset", data_dir, pcd_path)
            else:
                abs_path = pcd_path
            return abs_path

        def tokenize_add_label(sample):
            prompt_prefix = tokenizer.encode(tokenizer.bos_token + sample["prompt_prefix"], add_special_tokens=False)
            shape_pad_tokens = [0] * num_shape_tokens
            prompt_suffix = tokenizer.encode(sample["prompt_suffix"], add_special_tokens=False)
            code = tokenizer.encode(sample["code"] +  tokenizer.eos_token, add_special_tokens=False)
            input_ids = prompt_prefix + shape_pad_tokens + prompt_suffix + code
            shape_token_index = [len(prompt_prefix),  len(prompt_prefix)+len(shape_pad_tokens)]
            #[0] * len(prompt_prefix) + [1] * len(shape_pad_tokens) + [0] * (len(prompt_suffix)+len(code))
            # if not 'point_cloud' in sample["pcd_path"]:
            #     # part dataset convention
            #     # it could record relative path or absolute path
            #     # abs_path = os.path.join(current_path, "shape2code_dataset", data_dir, 'point_cloud', sample["pcd_path"])
            #     if dataset_config.pcd_path_prefix is None:
            #         abs_path = os.path.join(current_path, "shape2code_dataset", data_dir, 'point_cloud', sample["pcd_path"])
            #     else:
            #         abs_path = os.path.join(dataset_config.pcd_path_prefix, os.path.split(data_dir)[-1], 'point_cloud', sample["pcd_path"])
            #     sample["pcd_path"] = abs_path
            # elif 'point_cloud' in sample["pcd_path"] and not dataset_config.pcd_path_prefix is None:
            #     # object dataset convention
            #     abs_path = os.path.join(dataset_config.pcd_path_prefix, os.path.split(data_dir)[-1], sample["pcd_path"])
            #     sample["pcd_path"] = abs_path
            if isinstance(sample["pcd_path"], list):
                for i in range(len(sample["pcd_path"])):
                    sample["pcd_path"][i] = process_pcd_path(sample["pcd_path"][i], current_path, data_dir, dataset_config.pcd_path_prefix)
                if not dataset_config.max_cd_loss is None and split == 'train':
                    # only filter out large cd loss code and pcd pairs in the training set
                    if not sample['cd_loss'] is None:
                        assert len(sample["pcd_path"]) == len(sample['cd_loss'])
                        filtered_pcd_path = []
                        filtered_cd_loss = []
                        for i in range(len(sample["pcd_path"])):
                            if sample['cd_loss'][i] <= dataset_config.max_cd_loss:
                                filtered_pcd_path.append(sample["pcd_path"][i])
                                filtered_cd_loss.append(sample['cd_loss'][i])
                        sample["pcd_path"] = filtered_pcd_path
                        sample['cd_loss'] = filtered_cd_loss
            else:
                sample["pcd_path"] = process_pcd_path(sample["pcd_path"], current_path, data_dir, dataset_config.pcd_path_prefix)

            result = {
                "input_ids": input_ids,
                "attention_mask" : [1] * len(input_ids),
                "labels": [-100] * (len(prompt_prefix)+num_shape_tokens+len(prompt_suffix)) + code,
                "pcd_path": sample["pcd_path"],
                "shape_token_index": shape_token_index,
                'cd_loss': sample['cd_loss']
                }

            return result

        if isinstance(dataset[0]['pcd_path'], list):
            total_number_of_pcd_before_filter = count_pcd_number(dataset, num_proc=dataset_config.num_proc)
        dataset = dataset.map(tokenize_add_label, remove_columns=['prompt_prefix', 'prompt_suffix', 'code'], num_proc=dataset_config.num_proc)
        if isinstance(dataset[0]['pcd_path'], list):
            total_number_of_pcd_after_filter = count_pcd_number(dataset, num_proc=dataset_config.num_proc)
            if not dataset_config.max_cd_loss is None:
                print('pcd path is a list and we filter out samples with cd loss larger than %.4f' % dataset_config.max_cd_loss, flush=True)
                print('%.4f percent [%d/%d] point clouds remained after filtering large cd loss' % 
                        (total_number_of_pcd_after_filter/total_number_of_pcd_before_filter, 
                        total_number_of_pcd_after_filter, total_number_of_pcd_before_filter ), flush=True)

        if not dataset_labels is None:
            dataset = dataset.add_column("dataset_label", [dataset_labels[idx]]*len(dataset))
        dataset_list.append(dataset)
    dataset = datasets.concatenate_datasets(dataset_list)

    # filter dataset
    original_len = len(dataset)
    if not dataset_config.max_sequence_length is None and split == 'train':
        print('filtering out samples with more than %d input ids' % dataset_config.max_sequence_length, flush=True)
        dataset = dataset.filter(lambda example: len(example['input_ids'])<=dataset_config.max_sequence_length, num_proc=dataset_config.num_proc)
        print('%.4f percent [%d/%d] samples remained after filtering long sequence' % (len(dataset)/original_len, len(dataset), original_len ), flush=True)
    
    if not dataset_config.max_cd_loss is None and split == 'train':
        if 'cd_loss' in dataset.features.keys() and (not dataset[0]['cd_loss'] is None):
            print('filtering out samples with cd loss larger than %.4f' % dataset_config.max_cd_loss, flush=True)
            if isinstance(dataset[0]['cd_loss'], list):
                dataset = dataset.filter(lambda example: len(example['cd_loss'])>0, num_proc=dataset_config.num_proc)
            else:
                dataset = dataset.filter(lambda example: example['cd_loss']<=dataset_config.max_cd_loss, num_proc=dataset_config.num_proc)
            print('%.4f percent [%d/%d] samples remained after filtering large cd loss' % (len(dataset)/original_len, len(dataset), original_len ), flush=True)
    
    need_to_remove_keys = []
    for key in dataset.features.keys():
        # some key may be missing from the provided json file
        # in this case the dataset loader will give them none values, we need to remove them
        if dataset[0][key] is None:
            need_to_remove_keys.append(key)
            print('feature %s will be removed from the dataset' % key, flush=True)
    if len(need_to_remove_keys) > 0:
        dataset = dataset.remove_columns(need_to_remove_keys)
    # if dataset_config.extract_numbers_and_object_type:
    #     if dataset_config.dataset_version == 'single_object':
    #         dataset = dataset.filter(lambda example: len(example["numbers"])==6)
    # dataset = dataset.select(range(100))
    dataset = DatasetWrapper(dataset, num_points=dataset_config.num_points, 
                augmentation_args=dataset_config.augmentation_args)

    return dataset

if __name__ == '__main__':
    s = 'train: 10000, 10000, 10000; val: 2000, 2000, 2000; test: 2000, 2000, 2000'
    result = string_to_dict(s)
    pdb.set_trace()