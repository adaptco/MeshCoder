# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Blender dataset."""


import json

import py7zr

import datasets
import os
import numpy as np

import pdb


_CITATION = """
None
"""

_DESCRIPTION = """
SAMSum Corpus contains over 16k chat dialogues with manually annotated
summaries.
There are two features:
  - dialogue: text of dialogue.
  - summary: human written summary of the dialogue.
  - id: id of a example.
"""

_HOMEPAGE = "https://arxiv.org/abs/1911.12237"

_LICENSE = "CC BY-NC-ND 4.0"

# _URL = "https://huggingface.co/datasets/samsum/resolve/main/data/corpus.7z"
# set to local path by zylyu
# _URL = "data/corpus.7z"
# _URL = "data/without_rotation"


class Shape2Code(datasets.GeneratorBasedBuilder):
    """Instuct2Code Corpus dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="shape2code"),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("int32"),
                "prompt": datasets.Value("string"),
                "script": datasets.Value("string"),
                # "pcd_path": datasets.Value("string"),
                # "cd_loss": datasets.Value("float32"),
                "pcd_path": datasets.Sequence(datasets.Value("string")),
                "cd_loss": datasets.Sequence(datasets.Value("float32")),
                "category": datasets.Value("string"),
                "num_start": datasets.Sequence(datasets.Value("int32")),
                "num_end": datasets.Sequence(datasets.Value("int32")),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # path = dl_manager.download(_URL)
        path = dl_manager.download(self.config.data_dir)
        # pdb.set_trace()
        # path = os.path.join(os.path.realpath(__file__), 'data/corpus.7z')
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": (path, "train.json"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": (path, "test.json"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": (path, "val.json"),
                    "split": "val",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        path, fname = filepath
        with open(os.path.join(path, fname), "r") as f:
            data = json.load(f)
        for idx, example in enumerate(data):
            # example['pcd_path'] = os.path.join(path, example['pcd_path'])
            # yield example["id"], example
            yield idx, example
