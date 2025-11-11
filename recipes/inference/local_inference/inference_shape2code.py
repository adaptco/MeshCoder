# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time
import gradio as gr

import torch
from transformers import LlamaTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker, AgentType
from llama_recipes.inference.model_utils import load_model, load_peft_model
from llama_recipes.custom_models.modules.shape_tokenizer import ShapeTokenizer

from llama_recipes.configs.datasets import shape2code_dataset
from llama_recipes.datasets.shape_to_code_dataset import get_preprocessed_shape2code
# from transformers.data import DataCollatorForSeq2Seq
from llama_recipes.datasets.data_collator import CustomDataCollatorForSeq2Seq
from llama_recipes.utils.train_utils_shape2code import update_embeds, shape_to_text
from llama_recipes.utils.miscellaneous import batch_save_mesh, batch_save_pcd

# from llama_recipes.blender_scripts.code_to_mesh import batch_code_to_mesh
# from pytorch3d.loss import chamfer_distance

import yaml
import numpy as np
from accelerate.utils import is_xpu_available

from tqdm import tqdm

import pdb



def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=0, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    enable_llamaguard_content_safety: bool=False,
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    

    def inference(user_prompt, temperature, top_p, top_k, max_new_tokens, **kwargs,):
        safety_checker = get_safety_checker(enable_azure_content_safety,
                                            enable_sensitive_topics,
                                            enable_salesforce_content_safety,
                                            enable_llamaguard_content_safety
                                            )

        # Safety check of the user prompt
        # safety_results = [check(user_prompt) for check in safety_checker]
        # are_safe = all([r[1] for r in safety_results])
        are_safe = True
        if are_safe:
            print("User prompt deemed safe.")
            print(f"User prompt:\n{user_prompt}")
        else:
            print("User prompt deemed unsafe.")
            for method, is_safe, report in safety_results:
                if not is_safe:
                    print(method)
                    print(report)
            print("Skipping the inference as the prompt is not safe.")
            sys.exit(1)  # Exit the program with an error status

        # Set the seeds for reproducibility
        if is_xpu_available():
            torch.xpu.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        only_test_dataloader = False

        if not only_test_dataloader:
            model = load_model(model_name, quantization, use_fast_kernels)
            if peft_model:
                model = load_peft_model(model, peft_model)

            model.eval()

        # load shape tokenizer model
        with open(os.path.join(peft_model, 'config_shape_tokenizer.yml'), 'r') as yaml_file:
            shape_tokenizer_config = yaml.safe_load(yaml_file)
        
        shape_tokenizer_model = ShapeTokenizer(**shape_tokenizer_config)
        shape_tokenizer_ckpt = torch.load(os.path.join(peft_model, 'shape_tokenizer.pt'))
        shape_tokenizer_model.load_state_dict(shape_tokenizer_ckpt['model_state_dict'])
        if not only_test_dataloader:
            if next(model.parameters()).is_cuda:
                shape_tokenizer_model.cuda()
        shape_tokenizer_model.eval()

        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # shape2code_dataset
        with open(os.path.join(peft_model, 'config_dataset.yml'), 'r') as yaml_file:
            dataset_config = yaml.safe_load(yaml_file)
        
        # pdb.set_trace()
        # dataset_config['data_dir'] = 'data/0604dataset_primitive_100w'
        shape2code_dataset_config = shape2code_dataset(**dataset_config)
        # shape2code_dataset.num_shape_tokens = shape_tokenizer_config['querier_args']['num_query_tokens'] 
        # pdb.set_trace()
        # shape2code_dataset_config.dataset_labels = 'translation'
        dataset = get_preprocessed_shape2code(shape2code_dataset_config, tokenizer, 'test')
        # pdb.set_trace()
        # sample = dataset.__getitem__(10)
        batch_size = 32 if only_test_dataloader else 16
        num_workers = 0 if only_test_dataloader else 16
        dataloader = torch.utils.data.DataLoader(dataset, 
            collate_fn=CustomDataCollatorForSeq2Seq(tokenizer), 
            batch_size=batch_size, num_workers=num_workers, shuffle=False)
        
        start_idx = 10
        for i, batch in tqdm(enumerate(dataloader)):
            if only_test_dataloader:
                pdb.set_trace()
                # pass
            elif i*batch_size >=start_idx:
                # points = batch['points']
                # points = points[0].numpy()
                # np.savetxt('points.xyz', points)
                
                if is_xpu_available():
                    batch = {k: v.to("xpu") for k, v in batch.items() if isinstance(v, torch.Tensor)}
                else:
                    batch = {k: v.to("cuda") for k, v in batch.items() if isinstance(v, torch.Tensor)}
                instruction = batch['labels'][0] == (-100)
                code_start_index = instruction.long().sum()
                instruction_text = tokenizer.decode(batch['input_ids'][0,0:code_start_index], skip_special_tokens=False)
                code_gt_text = tokenizer.decode(batch['input_ids'][0,code_start_index:], skip_special_tokens=False)
                # print('gt code is', code_gt_text)
                # inputs_embeds = model.get_input_embeddings()(batch['input_ids']) # B, seq_len, embed_dim
                # batch = update_embeds(inputs_embeds, shape_tokenizer_model, batch)
                
                # batch['inputs_embeds'] = batch['inputs_embeds'][:,0:code_start_index,:]
                # batch['attention_mask'] = batch['attention_mask'][:,0:code_start_index]
                # if os.path.exists('inputs_embeds.pt'):
                #     saved_data = torch.load('inputs_embeds.pt')
                # else:
                #     torch.save(batch['inputs_embeds'], 'inputs_embeds.pt')
                # batch.pop('labels')
                # pdb.set_trace()

                start = time.perf_counter()

                output_result = shape_to_text(model, shape_tokenizer_model, tokenizer,
                    batch['points'], points_mask=batch['points_mask'], gt_points=batch['gt_points'],
                    eval_reconstruction_performance=True,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    min_length=min_length,
                    use_cache=use_cache,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    other_kwargs=kwargs)
                output_text = output_result['output_text_list']
                output_text = output_text[0]
                e2e_inference_time = (time.perf_counter()-start)*1000
                print(f"the inference time is {e2e_inference_time} ms")
                
                print('the %d-th sample' % i)
                print('gt code is:\n', code_gt_text)

                # Safety check of the model output
                # safety_results = [check(output_text, agent_type=AgentType.AGENT, user_prompt=user_prompt) for check in safety_checker]
                # are_safe = all([r[1] for r in safety_results])
                are_safe = True
                if are_safe:
                    print("User input and model output deemed safe.")
                    print(f"Model output:\n{output_text}")
                    print('\ncode to mesh success is', output_result['success'])
                    print('\ncd loss is', output_result['cd_loss'])
                else:
                    print("Model output deemed unsafe.")
                    for method, is_safe, report in safety_results:
                        if not is_safe:
                            print(method)
                            print(report)
                
                suffix = ['reconstruction_cd_%.5f' % cd for cd in output_result['cd_loss']]
                batch_save_mesh(output_result['verts'], output_result['faces'], 'vis', suffix, 
                        start_idx=i*batch_size, success=output_result['success'])
                batch_save_pcd(batch['gt_points'], 'vis', 'gt', start_idx=i*batch_size, success=None)
                batch_save_pcd(batch['points'], 'vis', 'input', start_idx=i*batch_size, success=None)
                pdb.set_trace()
        return output_text

    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"
        with open(prompt_file, "r") as f:
        #   user_prompt = "\n".join(f.readlines())
            user_prompt = "".join(f.readlines())
        inference(user_prompt, temperature, top_p, top_k, max_new_tokens)
    elif not sys.stdin.isatty():
        user_prompt = "\n".join(sys.stdin.readlines())
        inference(user_prompt, temperature, top_p, top_k, max_new_tokens)
    else:
        gr.Interface(
        fn=inference,
        inputs=[
            gr.components.Textbox(
                lines=9,
                label="User Prompt",
                placeholder="none",
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=1.0, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=1.0, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=50, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=200, label="Max tokens"
            ),
        ],
        outputs=[
            gr.components.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="Llama2 Playground",
        description="https://github.com/facebookresearch/llama-recipes",
        ).queue().launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    '''
    python inference.py --model_name ../../../llama-2-7b --peft_model --prompt_file prompt.txt --use_auditnlg
    '''
    fire.Fire(main)
