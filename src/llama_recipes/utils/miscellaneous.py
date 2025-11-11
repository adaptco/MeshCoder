import torch.distributed as dist
import torch
import trimesh
import os
import numpy as np

import pdb

def print_message(enable_fsdp, rank, message):
    if enable_fsdp:
        new_message = 'rank %d: ' % rank + message
    else:
        new_message = message
    print(new_message, flush=True)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='', world_size=1):
        self.reset()
        # name is the name of the quantity that we want to record, used as tag in tensorboard
        self.name = name
        self.world_size = world_size
    def reset(self):
        # self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1, summary_writer=None, global_step=None):
        # self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0
        if not summary_writer is None:
            # record the val in tensorboard
            summary_writer.add_scalar(self.name, val, global_step=global_step)
    
    def tensor_reduce(self):
        if not isinstance(self.sum, torch.Tensor):
            sums = torch.Tensor([self.sum]).cuda()
        else:
            sums = self.sum.cuda()
        if not isinstance(self.count, torch.Tensor):
            count = torch.Tensor([self.count]).cuda()
        else:
            count = self.count.cuda()

        if self.world_size > 1:
            dist.reduce(sums, 0, op=dist.ReduceOp.SUM)
            dist.reduce(count, 0, op=dist.ReduceOp.SUM)
        avg = sums / count
        return sums, count, avg
    
    def all_reduce(self):
        if not isinstance(self.sum, torch.Tensor):
            self.sum = torch.Tensor([self.sum]).cuda()
        if not isinstance(self.count, torch.Tensor):
            self.count = torch.Tensor([self.count]).cuda()
        dist.all_reduce(self.sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.count, op=dist.ReduceOp.SUM)
        if self.count.detach().cpu().item() > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0

class MultiAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, save_individual_values=False, initial_keys=None):
        global_average_meter = AverageMeter()
        self.average_meters = {'global':global_average_meter}
        if not initial_keys is None:
            for key in initial_keys:
                self.average_meters[key] = AverageMeter()
        self.save_individual_values = save_individual_values
        if save_individual_values:
            self.values = np.array([])
            self.types = np.array([])
        else:
            self.values = None
            self.types = None
        self.reset()

    def reset(self):
        for key in self.average_meters.keys():
            self.average_meters[key].reset()

    def update(self, values, types=None, max_value=None):
        # values is a tensor or numpy array of shape B
        # types is a list or numpy array of strings of length B
        # max_value is the maximum valid value in values, 
        # we will exclude non valid values when computing mean to avoid extreme values affecting mean
        # but note that those non valid values will still be recorded in the individual_values
        if not types is None:
            if isinstance(types, list):
                types = np.array(types)

        if max_value is None:
            filtered_values = values
            filtered_types = types
        else:
            valid = values <= max_value
            filtered_values = values[valid]
            if isinstance(valid, torch.Tensor):
                valid = valid.detach().cpu().numpy()
            filtered_types = None if types is None else types[valid]
        
        B = filtered_values.shape[0]
        if B>0:
            self.average_meters['global'].update(filtered_values.mean(), n=B)
            if not filtered_types is None:
                unique_types = np.unique(filtered_types)
                for key in unique_types:
                    if not key in self.average_meters.keys():
                        self.average_meters[key] = AverageMeter()
                        self.average_meters[key].reset()
                    
                    current_type = filtered_types==key
                    type_values = filtered_values[current_type]
                    self.average_meters[key].update(type_values.mean(), n=current_type.astype(int).sum())
        
        B = values.shape[0]
        if B>0:
            if self.save_individual_values:
                if isinstance(values, torch.Tensor):
                    values_array = values.detach().cpu().numpy()
                else:
                    values_array = values
                if self.values is None:
                    self.values = values_array
                else:
                    self.values = np.concatenate([self.values, values_array])
                
                if not types is None:
                    if self.types is None:
                        self.types = types
                    else:
                        self.types = np.concatenate([self.types, types])
            
    
    def all_reduce(self):
        for key in self.average_meters.keys():
            self.average_meters[key].all_reduce()
    
    def obtain_values(self, save_individual_values=False):
        result = {}
        for key in self.average_meters.keys():
            result[key] = self.average_meters[key].avg
            if isinstance(result[key], torch.Tensor):
                result[key] = result[key].detach().cpu().item()
        if save_individual_values:
            if not self.values is None:
                result['values'] = self.values.tolist()
            if not self.types is None:
                result['types'] = self.types.tolist()
        return result


def batch_save_mesh(verts_list, faces_list, save_dir, save_suffix, start_idx=0, success=None):
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    except Exception as error:
        print('an error occured when making directory:\n', save_dir)
        print('the error is:\n', error)
        print('continue without stop')
    num_meshes = len(verts_list)
    if success is None:
        idx = torch.arange(num_meshes)
    else:
        # in this case success is a tensor of shape num_samples, and success.sum() = 
        assert success.long().sum().detach().cpu().item() == num_meshes
        idx = torch.arange(success.shape[0])
        idx = idx[success.detach().cpu()>0]
    idx = idx + start_idx
    if isinstance(save_suffix, str):
        save_suffix = [save_suffix] * num_meshes
    for i in range(num_meshes):
        save_name = 'sample_%d_%s.ply' % (idx[i], save_suffix[i])
        if isinstance(verts_list[i], torch.Tensor):
            verts_list[i] = verts_list[i].detach().cpu().numpy()
        if isinstance(faces_list[i], torch.Tensor):
            faces_list[i] = faces_list[i].detach().cpu().numpy()
        mesh_save = trimesh.Trimesh(vertices=verts_list[i], faces=faces_list[i])
        mesh_save.export(os.path.join(save_dir, save_name))


def batch_save_pcd(points, save_dir, save_suffix, start_idx=0, success=None):
    # points is a tensor of shape B,N,3 or B,N,6
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    except Exception as error:
        print('an error occured when making directory:\n', save_dir)
        print('the error is:\n', error)
        print('continue without stop')
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    num_pcd = points.shape[0]
    if success is None:
        idx = torch.arange(num_pcd)
    else:
        # in this case success is a tensor of shape num_samples, and success.sum() = 
        assert success.long().sum().detach().cpu().item() == num_pcd
        idx = torch.arange(success.shape[0])
        # pdb.set_trace()
        idx = idx[success.detach().cpu()>0]
    idx = idx + start_idx
    if isinstance(save_suffix, str):
        save_suffix = [save_suffix] * num_pcd
    for i in range(num_pcd):
        save_name = 'sample_%d_%s.xyz' % (idx[i], save_suffix[i])
        np.savetxt(os.path.join(save_dir, save_name), points[i])


def batch_write_code_list_to_file(code_list, save_dir, save_suffix, start_idx=0):
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    except Exception as error:
        print('an error occured when making directory:\n', save_dir)
        print('the error is:\n', error)
        print('continue without stop')
    if isinstance(save_suffix, str):
        save_suffix = [save_suffix] * len(code_list)
    for i in range(len(code_list)):
        save_name = 'sample_%d_%s.py' % (i+start_idx, save_suffix[i])
        f = open(os.path.join(save_dir, save_name), "w")
        f.write(code_list[i])
        f.close()

def partial_load(model, pretrained_dict, allow_shape_mismatch=False):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    if allow_shape_mismatch:
        match_num = 0
        mis_match_num = 0
        for key in pretrained_dict.keys():
            if key in model_dict.keys() and model_dict[key].shape == pretrained_dict[key].shape:
                model_dict[key] = pretrained_dict[key]
                match_num = match_num + 1
            else:
                mis_match_num = mis_match_num + 1
        total_num = match_num + mis_match_num
        print('[%d|%d] %.4f percent paramters matches' % (match_num, total_num, match_num/total_num*100), flush=True)
    else:
        model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model