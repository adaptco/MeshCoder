import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

import pdb

def pad_list_to_max_length(input_list, pad_number=0, return_idx=True):
    # input_list is a list of list, each element list contain numbers
    length = [len(l) for l in input_list]
    max_length = max(length)
    if return_idx:
        idx_list = [torch.arange(max_length)<lg for lg in length]
        idx = torch.stack(idx_list, dim=0)
    
    new_list = [(l+[pad_number]*(max_length-len(l))) for l in input_list]
    output = torch.Tensor(new_list)
    if return_idx:
        return output, idx
    else:
        return output

def pad_tensor(tensor_list, pad_value=0):
    # tensor_list is a list of tensors, each tensor is of shape seq_len, ...
    for i in range(len(tensor_list)):
        if isinstance(tensor_list[i], np.ndarray):
            tensor_list[i] = torch.from_numpy(tensor_list[i]).float()
    lengths = [tensor.shape[0] for tensor in tensor_list]
    max_length = max(lengths)
    # make sure at least max_length has length 1
    if max_length < 1:
        print('max length less than 1')
        for kk in range(len(tensor_list)):
            print(kk, tensor_list[kk].shape)
        
    max_length = max(max_length, 1)
    pad_length = [max_length-length for length in lengths]
    # padded_tensor = torch.ones_like(tensor_list[0][0]) * pad_value # of shape ...
    padded_tensor = torch.ones(tensor_list[0].shape[1:], dtype=tensor_list[0].dtype, device=tensor_list[0].device) * pad_value 
    # of shape ...
    pad_tensor_list = [torch.stack([padded_tensor]*pad_length[i]) if pad_length[i]>0 
                        else None for i in range(len(tensor_list))]
    new_tensor_list = [torch.cat([tensor_list[i], pad_tensor_list[i]]) if pad_length[i]>0 
                        else tensor_list[i] for i in range(len(tensor_list))]
    output = torch.stack(new_tensor_list) # B, max_length, ...
    indices = torch.arange(max_length).unsqueeze(0) # 1, max_length
    lengths = torch.Tensor(lengths).unsqueeze(1) # B, 1
    mask = indices < lengths # B, max_length
    return output, mask

def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

@dataclass
class CustomDataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        # features is a list of length batchsize
        # each element is a sample from the dataset
        # in our case, features[0].keys() has 'numbers', 'input_ids', 'attention_mask', 'labels', 
        # 'shape_token_index', 'points', 'dataset_label'
        contain_numbers = False
        if 'numbers' in features[0].keys():
            contain_numbers = True
            numbers_list = [fea.pop('numbers') for fea in features]
            numbers_tensor, numbers_idx = pad_list_to_max_length(numbers_list, pad_number=0, return_idx=True)
        
        possible_strings = ['dataset_label', 'pcd_path', 'category']
        string_dict = {}
        for key in possible_strings:
            if key in features[0].keys():
                string_dict[key] = [fea.pop(key) for fea in features]

        # dataset_label = None
        # if 'dataset_label' in features[0].keys():
        #     dataset_label = [fea.pop('dataset_label') for fea in features]
        
        # pcd_path = None
        # if 'pcd_path' in features[0].keys():
        #     pcd_path = [fea.pop('pcd_path') for fea in features]
        
        points_list = [fea.pop('points') for fea in features]
        try:
            points_tensor, points_mask = pad_tensor(points_list, pad_value=0)
        except:
            print(points_list)
            print(pcd_path)
            for kk in range(len(points_list)):
                print(kk, points_list[kk].shape)
            pdb.set_trace()
        
        features = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        if contain_numbers:
            features['numbers'] = numbers_tensor
            features['numbers_idx'] = numbers_idx
        # if not dataset_label is None:
        #     features['dataset_label'] = dataset_label
        # if not pcd_path is None:
        #     features['pcd_path'] = pcd_path
        for key in string_dict:
            features[key] = string_dict[key]
        features['points'] = points_tensor
        features['points_mask'] = points_mask
        return features