# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging
from datetime import datetime
import traceback

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json
import numpy as np

import time

from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint, save_peft_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from llama_recipes.utils.miscellaneous import AverageMeter, MultiAverageMeter
from accelerate.utils import is_xpu_available, is_ccl_available
from torch.nn.parallel import DistributedDataParallel
from pytorch3d.loss import chamfer_distance
from llama_recipes.blender_scripts.code_to_mesh import batch_code_to_mesh
from llama_recipes.utils.miscellaneous import batch_save_mesh, batch_save_pcd, batch_write_code_list_to_file

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from llama_recipes.custom_models.modeling_llama_3_2 import LlamaForCausalLM
from peft import PeftModel

import pdb

def prepare_saving_peft_ckpt(save_dir, enable_fsdp, rank):
    if not enable_fsdp or rank == 0:
        readme_path = os.path.join(save_dir, 'README.md')
        if not os.path.exists(readme_path):
            os.makedirs(save_dir, exist_ok=True)
            f = open(readme_path, "w")
            f.write("Now the file has some content!")
            f.close()
        
        adapter_path = os.path.join(save_dir, 'adapter_model.safetensors')
        if os.path.exists(adapter_path):
            os.remove(adapter_path)

    if enable_fsdp:
        dist.barrier()

def update_embeds(inputs_embeds, shape_tokenizer_model, batch):
    shape_tokens = shape_tokenizer_model(batch['points'], mask=batch.get('points_mask', None)) # B, num_shape_tokens, embed_dim
    # print('data device:', batch['points'].device, 'tokenizer device:', next(shape_tokenizer_model.parameters()).device)
    if shape_tokens.abs().max() > 100:
        print('shape_tokens min and max:', shape_tokens.min(), shape_tokens.max())
    # inputs_embeds = model.get_input_embeddings()(batch['input_ids']) # B, seq_len, embed_dim
    B, seq_len, embed_dim = inputs_embeds.shape[0], inputs_embeds.shape[1], inputs_embeds.shape[2]
    start = batch['shape_token_index'][0][0]
    end = batch['shape_token_index'][0][1]
    device = inputs_embeds.device
    shape_tokens_padded = torch.cat([
        torch.zeros(B, start, embed_dim).to(device),
        shape_tokens,
        torch.zeros(B, seq_len-end, embed_dim).to(device),
        ], dim=1)
    # B, seq_len, embed_dim

    shape_token_index = torch.zeros(1, seq_len, 1).to(device)
    shape_token_index[:,start:end,:] = 1
    inputs_embeds_updated = inputs_embeds * (1-shape_token_index) + shape_tokens_padded * shape_token_index
    batch['inputs_embeds'] = inputs_embeds_updated
    # batch.pop('points')
    # batch.pop('shape_token_index')
    # batch.pop('input_ids')
    return batch

def obtain_normalize_parameters(pcd):
    # pcd is of shape B,N,3
    minn = torch.min(pcd, dim=1, keepdim=True)[0] # B,1,3
    maxx = torch.max(pcd, dim=1, keepdim=True)[0] # B,1,3
    center = (maxx+minn) / 2 # B,1,3
    max_length = torch.max(maxx-minn, dim=2, keepdim=True)[0] # B,1,1
    return center, max_length

def normalize_pair_pcd(gt, pred):
    # gt and pred is of shape B,N,3
    # normalize each pcd to [-1,1]^3
    center, max_length = obtain_normalize_parameters(gt)
    max_length = torch.clamp(max_length, min=1e-6)
    new_gt = (gt - center) / max_length * 2
    new_pred = (pred - center) / max_length * 2
    return new_gt, new_pred

def shape_to_text(model, shape_tokenizer_model, tokenizer,
                    points, points_mask=None, gt_points=None, eval_reconstruction_performance=False,
                    max_new_tokens=100,
                    do_sample=True,
                    top_p=1.0,
                    temperature=1.0,
                    min_length=0,
                    use_cache=True,
                    top_k=50,
                    repetition_penalty=1.0,
                    length_penalty=1,
                    other_kwargs={}):
    # points of shape B,N,3 or B,N,6
    prompt_prefix = "Write Blender Python script according to the following instructions:\n"
    prompt_suffix = "\n---\nCode:\n"
    prompt_prefix = tokenizer.encode(tokenizer.bos_token + prompt_prefix, add_special_tokens=False)
    # shape_pad_tokens = [0] * num_shape_tokens
    prompt_suffix = tokenizer.encode(prompt_suffix, add_special_tokens=False)

    device = points.device
    shape_tokens = shape_tokenizer_model(points, mask=points_mask) # B, num_shape_tokens, embed_dim
    B = shape_tokens.shape[0]
    prompt_prefix = torch.stack( [torch.Tensor(prompt_prefix).long().to(device)]*B, dim=0) # B, prefix_len
    prompt_suffix = torch.stack( [torch.Tensor(prompt_suffix).long().to(device)]*B, dim=0) # B, suffix_len

    # for fsdp to work, we need to use forward
    prompt_prefix_embed = model(input_ids=prompt_prefix, only_return_inputs_embeds=True) # B, prefix_len, embed_dim
    prompt_suffix_embed = model(input_ids=prompt_suffix, only_return_inputs_embeds=True) # B, suffix_len, embed_dim

    inputs_embeds = torch.cat([prompt_prefix_embed, shape_tokens, prompt_suffix_embed], dim=1)
    # pdb.set_trace()
    output_text_list = generate_text(model, tokenizer, inputs_embeds, attention_mask=None, 
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    min_length=min_length,
                    use_cache=use_cache,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    other_kwargs=other_kwargs)
    result = {'output_text_list':output_text_list}
    if eval_reconstruction_performance:
        # num_points = points.shape[1]
        if gt_points is None:
            # sometimes points may contain data augmentation
            gt_points = points
        num_points = gt_points.shape[1]
        mesh_and_pcd = batch_code_to_mesh(output_text_list, 
                'cache', 'test.py', device=device, sample_points=True, num_points=num_points, execute_method='exec')
        result['success'] = mesh_and_pcd['success']
        result['points'] = mesh_and_pcd.get('points', torch.zeros(0,1,3))
        result['verts'] = mesh_and_pcd.get('verts', [])
        result['faces'] = mesh_and_pcd.get('faces', [])
        result['error'] = mesh_and_pcd['error']
        if result['success'].sum() > 0:
            cd_loss = chamfer_distance(mesh_and_pcd['points'], gt_points[result['success']>0,:,0:3], batch_reduction=None)[0]
            result['cd_loss'] = cd_loss
            normalized_gt, normalized_pred = normalize_pair_pcd(gt_points[result['success']>0,:,0:3], mesh_and_pcd['points'])
            normalized_cd_loss = chamfer_distance(normalized_pred, normalized_gt, batch_reduction=None)[0]
            result['normalized_cd_loss'] = normalized_cd_loss
        else:
            # result['cd_loss'] = -1
            result['cd_loss'] = torch.zeros(0).to(device)
            result['normalized_cd_loss'] = torch.zeros(0).to(device)
    return result


def generate_text(model, tokenizer, inputs_embeds, attention_mask, 
                    max_new_tokens=100,
                    do_sample=True,
                    top_p=1.0,
                    temperature=1.0,
                    min_length=0,
                    use_cache=True,
                    top_k=50,
                    repetition_penalty=1.0,
                    length_penalty=1,
                    other_kwargs={}):
    
    with torch.no_grad():
        outputs = model.generate(
            # **batch,
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            min_length=min_length,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **other_kwargs)
    output_text_list = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # print(output_text_list[0])
    # pdb.set_trace()
    return output_text_list

def train_shape2code(model, shape_tokenizer_model, 
        train_dataloader, eval_dataloader, tokenizer, 
        optimizer, lr_scheduler, optimizer_shape_tokenizer, lr_scheduler_shape_tokenizer, 
        gradient_accumulation_steps, train_config, dataset_config, fsdp_config=None, 
        local_rank=None, rank=None, wandb_run=None, start_epoch=0):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"]) 

    

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []
        
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    if train_config.enable_fsdp:
        dist.barrier()
    for epoch in range(start_epoch, train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            shape_tokenizer_model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            torch.cuda.empty_cache()
            warm_up_steps = 0
            for step, batch in enumerate(train_dataloader):
                # pbar.update(1)
                # continue
                # batch.keys(): ['input_ids', 'attention_mask', 'labels']
                # batch['input_ids'].shape: [4, 4096], they are indices of each token
                # batch['labels'].shape: [4, 4096], they mostly the same as input_ids, but some components are set to -100
                # batch['attention_mask'].shape: [4, 4096]
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:
                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            else:
                                batch[key] = batch[key].to('cuda:0')
                    # print(key, batch[key].shape)
                    # if batching method is padding
                    # input_ids padded with 2, attention_mask padded with 0, labels padded with -100
                with autocast():
                    # inputs_embeds = model.get_input_embeddings()(batch['input_ids']) # B, seq_len, embed_dim
                    # for fsdp to work, we need to use the forward function
                    # print('rank:', rank, batch['input_ids'].shape)
                    inputs_embeds = model(input_ids=batch['input_ids'], only_return_inputs_embeds=True) # B, seq_len, embed_dim
                    # print(batch['input_ids'].shape, inputs_embeds.shape)
                    batch = update_embeds(inputs_embeds, shape_tokenizer_model, batch)
                    # some times the first step could cause cuda out of memory, therefore we use a small batchsize to warm up
                    if step >= warm_up_steps:
                        loss = model(inputs_embeds=batch['inputs_embeds'], labels=batch['labels'], attention_mask=batch['attention_mask']).loss
                    else:
                        loss = model(inputs_embeds=batch['inputs_embeds'][0:1], labels=batch['labels'][0:1], attention_mask=batch['attention_mask'][0:1]).loss
                # pdb.set_trace()
                loss = loss / gradient_accumulation_steps
                if train_config.save_metrics:
                    train_step_loss.append(loss.detach().float().item())
                    train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                total_loss += loss.detach().float()
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                            scaler.unscale_(optimizer)
                            if train_config.enable_fsdp:
                                model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                        if step >= warm_up_steps:
                            scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        pbar.update(1)
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                            if train_config.enable_fsdp:
                                model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                        if step >= warm_up_steps:
                            optimizer.step()
                        optimizer.zero_grad()
                        if epoch>=train_config.shape_tokenizer_warm_up_epochs:
                            if step >= warm_up_steps:
                                optimizer_shape_tokenizer.step()
                        optimizer_shape_tokenizer.zero_grad()
                        pbar.update(1)

                if wandb_run:
                    if not train_config.enable_fsdp or rank==0:
                        wandb_run.log({
                            'train/epoch': epoch + 1,
                            'train/step': epoch * len(train_dataloader) + step,
                            'train/loss': loss.detach().float(),
                            'train/lr-llama': lr_scheduler.get_last_lr()[0],
                            'train/lr-shape-tokenizer': lr_scheduler_shape_tokenizer.get_last_lr()[0],
                        })

                pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                if train_config.save_metrics:
                    save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
            pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)
        
        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))
        
        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()
        if epoch>=train_config.shape_tokenizer_warm_up_epochs:
            lr_scheduler_shape_tokenizer.step()

        if train_config.run_validation:
            eval_all_metrics, eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(
                model, shape_tokenizer_model, train_config, dataset_config, eval_dataloader, local_rank, rank, 
                tokenizer, wandb_run, epoch, visualization=train_config.visualization)
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)

            checkpoint_start_time = time.perf_counter()
            if train_config.shape_tokenizer_warm_up_epochs>0:
                if epoch == train_config.shape_tokenizer_warm_up_epochs-1:
                    original_output_dir = train_config.output_dir
                    train_config.output_dir = os.path.join(train_config.output_dir, 'warm_up_epoch_%d' % epoch)
                elif epoch == train_config.shape_tokenizer_warm_up_epochs:
                    train_config.output_dir = original_output_dir
                
            if train_config.save_model and eval_epoch_loss < best_val_loss:
                if train_config.enable_fsdp:
                    dist.barrier()
                # pdb.set_trace()
                if not train_config.enable_fsdp or rank == 0:
                    os.makedirs(train_config.output_dir, exist_ok=True)
                    save_path = os.path.join(train_config.output_dir, 'shape_tokenizer.pt')
                    torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': (shape_tokenizer_model.module.state_dict() 
                            if isinstance(shape_tokenizer_model, DistributedDataParallel) 
                            else shape_tokenizer_model.state_dict()),
                        'optimizer_state_dict': optimizer_shape_tokenizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler_shape_tokenizer.state_dict(),
                        'loss': eval_epoch_loss,
                        'eval_all_metrics': eval_all_metrics,
                        }, save_path)
                    print(f"shape tokenizer model checkpoint saved for epoch {epoch+1} at {save_path}\n")
                if train_config.use_peft:
                    if train_config.enable_fsdp:
                        if rank==0:
                            print(f"we are about to save the PEFT modules")
                    else:
                        print(f"we are about to save the PEFT modules")
                    # model.save_pretrained(train_config.output_dir)
                    # if train_config.enable_fsdp:
                    #     time.sleep(rank*10)
                    prepare_saving_peft_ckpt(train_config.output_dir, train_config.enable_fsdp, rank)
                    save_peft_checkpoint(model, train_config.output_dir, is_main_process=((not train_config.enable_fsdp) or rank==0))
                    if train_config.enable_fsdp:
                        if rank==0:
                            print(f"PEFT modules are saved in {train_config.output_dir} directory")
                    else:
                        print(f"PEFT modules are saved in {train_config.output_dir} directory")

                else:
                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:

                        save_model_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch
                        )
                    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                        print("=====================================================")

                        save_model_and_optimizer_sharded(model, rank, train_config)
                        if train_config.save_optimizer:
                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                            print("=====================================================")

                    if not train_config.use_peft and  train_config.save_optimizer:
                        save_optimizer_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch
                        )
                        print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                        print("=====================================================")
                if train_config.enable_fsdp:
                    dist.barrier()
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))
        if train_config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        
        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename

    #saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results

def evaluation(model, shape_tokenizer_model, train_config, dataset_config, eval_dataloader, local_rank, rank, 
                tokenizer, wandb_run, epoch, visualization=False):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1
    model.eval()
    shape_tokenizer_model.eval()
    # eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    # eval_loss = 0.0  # Initialize evaluation loss
    mertic_list = ['next_token_pred_loss', 'success', 'cd_loss', 'normalized_cd_loss']
    if not train_config.max_valid_cd_loss is None:
        mertic_list = mertic_list + ['cd_loss_valid_ratio', 'normalized_cd_loss_valid_ratio']
    metric_dict = {}
    dataset_labels = dataset_config.dataset_labels
    if not dataset_labels is None:
        # sometimes we observe it is already a tuple splitted by ,
        if isinstance(dataset_labels, str):
            dataset_labels = dataset_labels.split(',')
            dataset_labels = [dataset_label.strip() for dataset_label in dataset_labels]
    for key in mertic_list:
        # metric_dict[key] = AverageMeter()
        metric_dict[key] = MultiAverageMeter(save_individual_values=True, initial_keys=dataset_labels)

    # if not train_config.enable_fsdp or rank<8:
    # fsdp model has bugs when generate, therefore we save and then load a normal LlamaForCausalLM model
    print('saving model for evaluation, rank:', rank)
    # all ranks need to execute this command
    # if train_config.enable_fsdp:
    #     time.sleep(rank*10)
    prepare_saving_peft_ckpt(os.path.join(train_config.output_dir, 'eval_cache'), train_config.enable_fsdp, rank)
    save_peft_checkpoint(model, os.path.join(train_config.output_dir, 'eval_cache'), is_main_process=((not train_config.enable_fsdp) or rank==0))
    # model.save_pretrained(os.path.join(train_config.output_dir, 'eval_cache'))
    if train_config.enable_fsdp:
        dist.barrier()
    print('loading model for code generation evaluation, rank:', rank)
    model_for_generation = LlamaForCausalLM.from_pretrained(
        train_config.model_name,
        return_dict=True,
        load_in_8bit=train_config.quantization,
        device_map='cuda:%d' % local_rank, #"auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" if train_config.use_fast_kernels else None,
    )
    model_for_generation = PeftModel.from_pretrained(model_for_generation, os.path.join(train_config.output_dir, 'eval_cache'))
    model_for_generation.eval()
    
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    if train_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    else:
                        if is_xpu_available():
                            batch[key] = batch[key].to('xpu:0')
                        else:
                            batch[key] = batch[key].to('cuda:0')  
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                # for fsdp to work, we need to use the forward function
                inputs_embeds = model(input_ids=batch['input_ids'], only_return_inputs_embeds=True) # B, seq_len, embed_dim
                batch = update_embeds(inputs_embeds, shape_tokenizer_model, batch)
                outputs = model(inputs_embeds=batch['inputs_embeds'], labels=batch['labels'], attention_mask=batch['attention_mask'], loss_batch_reduction=False)
                loss = outputs.loss
                if train_config.save_metrics:
                    val_step_loss.append(loss.mean().detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.mean().detach().float())))  

                # update metrics
                bacth_num_samples = batch['input_ids'].shape[0]
                # metric_dict['next_token_pred_loss'].update(loss.detach().float(), n=bacth_num_samples)
                metric_dict['next_token_pred_loss'].update(loss.detach().float(), types=batch.get('dataset_label', None))

                # generate code, compare meshes from code and gt point cloud
                torch.cuda.empty_cache()
                try:
                    generation_result = shape_to_text(model_for_generation, shape_tokenizer_model, tokenizer,
                        batch['points'], points_mask=batch['points_mask'], gt_points=batch['gt_points'],
                        eval_reconstruction_performance=True,
                        max_new_tokens=1024 if dataset_config.max_sequence_length is None else dataset_config.max_sequence_length,
                        do_sample=False,
                        top_p=1.0,
                        temperature=1.0,
                        min_length=0,
                        use_cache=True,
                        top_k=50,
                        repetition_penalty=1.0,
                        length_penalty=1,
                        other_kwargs={})
                    metric_dict['success'].update(generation_result['success'].detach().float(), types=batch.get('dataset_label', None))
                    if 'dataset_label' in batch.keys() and generation_result['success'].sum() > 0:
                        cd_loss_types = np.array(batch['dataset_label'])
                        cd_loss_types = cd_loss_types[generation_result['success'].detach().cpu().numpy()>0]
                    else:
                        cd_loss_types = None
                    if generation_result['success'].sum() > 0:
                        metric_dict['cd_loss'].update(generation_result['cd_loss'].detach().float(), 
                                        types=cd_loss_types, max_value=train_config.max_valid_cd_loss)
                        metric_dict['normalized_cd_loss'].update(generation_result['normalized_cd_loss'].detach().float(), 
                                        types=cd_loss_types, max_value=train_config.max_valid_cd_loss)
                        # pdb.set_trace()
                        if not train_config.max_valid_cd_loss is None:
                            metric_dict['cd_loss_valid_ratio'].update(
                                        (generation_result['cd_loss']<=train_config.max_valid_cd_loss).detach().float(), 
                                        types=cd_loss_types)
                            metric_dict['normalized_cd_loss_valid_ratio'].update(
                                        (generation_result['normalized_cd_loss']<=train_config.max_valid_cd_loss).detach().float(), 
                                        types=cd_loss_types)

                    if visualization:
                        visualization_folder = os.path.join(train_config.output_dir, 'eval_cache/visualization_epoch_%d' % (epoch+1))
                        os.makedirs(visualization_folder, exist_ok=True)
                        start_idx = step*eval_dataloader.batch_size

                        suffix = ['rank_%d' % rank for kk in range(batch['points'].shape[0])]
                        useful_keys = ['dataset_label', 'category']
                        for key in useful_keys:
                            if key in batch.keys():
                                suffix = [(suffix[kk] + '_' + batch[key][kk]) for kk in range(batch['points'].shape[0])]

                        # save input pcd
                        input_pcd_suffix = [suff + '_input_pcd' for suff in suffix]
                        if 'pcd_path' in batch.keys():
                            assert len(batch['pcd_path']) == len(input_pcd_suffix)
                            for pcd_idx in range(len(input_pcd_suffix)):
                                current_pcd_path = batch['pcd_path'][pcd_idx].replace('/', '_')
                                current_pcd_path = current_pcd_path.replace('//', '_')
                                current_pcd_path = current_pcd_path.replace(':', '_')
                                if not current_pcd_path[0] == '_':
                                    current_pcd_path = '_' + current_pcd_path
                                input_pcd_suffix[pcd_idx] = input_pcd_suffix[pcd_idx] + current_pcd_path
                        # pdb.set_trace()
                        batch_save_pcd(batch['points'], visualization_folder, input_pcd_suffix, start_idx=start_idx, success=None)
                        
                        # save predicted code
                        pred_code_suffix = [suff + '_pred_code' for suff in suffix]
                        if 'error' in generation_result.keys():
                            for error_idx in range(len(generation_result['error'])):
                                if not generation_result['error'][error_idx] is None:
                                    pred_code_suffix[error_idx] = pred_code_suffix[error_idx] + '_error_happened'
                                    generation_result['output_text_list'][error_idx] = (generation_result['output_text_list'][error_idx] +
                                            '\n\nthe error is\n\n' + generation_result['error'][error_idx])
                        batch_write_code_list_to_file(generation_result['output_text_list'], visualization_folder, pred_code_suffix, start_idx=start_idx)
                        
                        # save gt code
                        gt_code = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                        # start_code_list = ['Write Blender Python script according to the following instructions:\n \n---\nCode:\n ', 'Write Blender Python script according to the following instructions:\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n---\nCode:\n']
                        # for start_code in start_code_list:
                        #     if start_code in gt_code[0]:
                        #         break
                        # gt_code = [code.split(start_code)[1] for code in gt_code]
                        gt_code_no_instruction = []
                        for code in gt_code:
                            code_split = code.split('\n---\nCode:\n')
                            if len(code_split) == 2:
                                gt_code_no_instruction.append(code_split[1])
                            else:
                                gt_code_no_instruction.append(code)
                        # gt_code = [code.split('\n---\nCode:\n')[1] for code in gt_code]
                        gt_code_suffix = [suff + '_gt_code' for suff in suffix]
                        batch_write_code_list_to_file(gt_code_no_instruction, visualization_folder, gt_code_suffix, start_idx=start_idx)

                        if generation_result['success'].sum() > 0:
                            success_suffix = np.array(suffix)[generation_result['success'].detach().cpu().numpy()>0]
                            cd_suffix = [success_suffix[kk] + '_cd_%.5f' % generation_result['cd_loss'][kk] for kk in range(generation_result['cd_loss'].shape[0])]
                            
                            pred_mesh_suffix = [suff + '_pred_mesh' for suff in cd_suffix]
                            pred_pcd_suffix = [suff + '_pred_pcd' for suff in cd_suffix]
                            batch_save_mesh(generation_result['verts'], generation_result['faces'], visualization_folder, pred_mesh_suffix, 
                                    start_idx=start_idx, success=generation_result['success'])
                            batch_save_pcd(generation_result['points'], visualization_folder, pred_pcd_suffix, start_idx=start_idx, success=generation_result['success'])

                except Exception as error:
                    # sometime an error will happen in model.generate such as cuda oom
                    # we continue eval other batches
                    print('an error occured while generating code')
                    print('the error is')
                    print(error)
                    print(traceback.format_exc())
                    torch.cuda.empty_cache()
                    # metric_dict['success'].update(0, n=bacth_num_samples)
                    metric_dict['success'].update(torch.zeros(bacth_num_samples).to(batch['points'].device),
                                                    types=batch.get('dataset_label', None))

            # Decode predictions and add to evaluation predictions list
            # preds = torch.argmax(outputs.logits, -1)
            # eval_preds.extend(
            #     tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            # )

    # print('\nbefore all reduce')
    # for key in metric_dict.keys():
    #     print('metric %s rank %d sum %.2f count %d' % (key, rank, 
    #         metric_dict[key].sum, metric_dict[key].count))
    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        for key in metric_dict.keys():
            metric_dict[key].all_reduce()
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        for key in metric_dict.keys():
            metric_dict[key].all_reduce()

    # print('\nafter all reduce')
    # for key in metric_dict.keys():
    #     print('metric %s rank %d sum %.2f count %d' % (key, rank, 
    #         metric_dict[key].sum, metric_dict[key].count))

    # prepare final evaluation metrics to print and report
    report_values = {}
    for key in metric_dict.keys():
        report_values['eval/'+key] = metric_dict[key].obtain_values()
    
    report_values['eval/perplexity'] = {}
    report_values['eval/cd_loss_divide_success'] = {}
    report_values['eval/normalized_cd_loss_divide_success'] = {}
    for key in report_values['eval/next_token_pred_loss'].keys():
        # sometimes at the initial training phase, success rate is low and cd loss could miss many values
        # there could be keys in report_values['eval/next_token_pred_loss'] that doesnot exist in
        # report_values['eval/cd_loss_divide_success']
        try:
            report_values['eval/perplexity'][key] = np.exp(report_values['eval/next_token_pred_loss'][key])
            report_values['eval/cd_loss_divide_success'][key] = report_values['eval/cd_loss'][key] / max(report_values['eval/success'][key], 0.001)
            report_values['eval/normalized_cd_loss_divide_success'][key] = report_values['eval/normalized_cd_loss'][key] / max(report_values['eval/success'][key], 0.001)
        except Exception as error:
            # sometime an error will happen in model.generate such as cuda oom
            # we continue eval other batches
            print('an error occured when computing perplexity and cd_loss_divide_success')
            print('the error is')
            print(error)
            print('key:', key)
            print('report_values:')
            print(json.dumps(report_values, indent=4, sort_keys=False))

    # pdb.set_trace()
    # Print evaluation metrics
    if not train_config.enable_fsdp or rank==0:
        print(json.dumps(report_values, indent=4, sort_keys=False), flush=True)

    if wandb_run: 
        wandb_run.log(report_values, commit=False)

    # clear cache
    del model_for_generation
    torch.cuda.empty_cache()
    
    # eval_epoch_loss = report_values['eval/cd_loss'] / max(report_values['eval/success'], 0.001)
    # we use cd_loss_divide_success to determine and save the best ckpt
    return report_values, report_values['eval/perplexity']['global'], report_values['eval/normalized_cd_loss_divide_success']['global'], val_step_loss, val_step_perplexity
    # return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity