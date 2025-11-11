# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import os
import pdb

def get_preprocessed_instuct2code(dataset_config, tokenizer, split):
    current_path = os.path.split(os.path.realpath(__file__))[0]
    # pdb.set_trace()
    dataset = datasets.load_dataset(os.path.join(current_path, "instuct2code_dataset"), split=split)

    prompt = (
        f"Write Blender Python script according to the following instructions:\n{{instruct}}\n---\nCode:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(instruct=sample["prompt"]),
            "code": sample["script"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        code = tokenizer.encode(sample["code"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + code,
            "attention_mask" : [1] * (len(prompt) + len(code)),
            "labels": [-100] * len(prompt) + code,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
