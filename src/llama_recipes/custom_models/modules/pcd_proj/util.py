import torch
from PIL import Image
import numpy as np
import os
import trimesh

import pdb

class EpochScheduler:

    def __init__(self, epoch_milestones, values, value_name='loss weight'):
        # epoch_milestones is like   [100, 200, 300, 400]
        # values           is like  [0,  0.1, 0.2, 0.5,  1]
        self.epoch_milestones = epoch_milestones
        self.epoch_milestones = [0] + self.epoch_milestones
        self.epoch_milestones.append(np.inf)

        self.values = values
        self.current_value = None
        self.value_name = value_name

    def obtain_value(self, epoch):
        # self.epoch_milestones is like [0, 100, 200, 300, 400, np.inf]
        # self.values           is like  [0,  0.1, 0.2, 0.5,  1]
        for i in range(len(self.epoch_milestones)):
            if epoch >= self.epoch_milestones[i] and epoch < self.epoch_milestones[i+1]:
                new_value = self.values[i]
                break
        if self.current_value is None:
            print('%s is initialized as %.4f at epoch %d' % (self.value_name, new_value, epoch), flush=True)
            self.current_value = new_value
        elif not new_value == self.current_value:
            print('%s is changed from %.4f to %.4f at epoch %d' % 
                    (self.value_name, self.current_value, new_value, epoch), flush=True)
            self.current_value = new_value
        return new_value

# Positional encoding
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : False,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def save_mesh(save_file, verts, faces):
    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(save_file)

def batch_save_images_same_dir(x, save_dir, name_format, start_idx=0):
    # x is a numpy array of shape BHW or BHW3 or BHW4
    os.makedirs(save_dir, exist_ok=True)
    for i in range(x.shape[0]):
        save_name = name_format % str(start_idx+i).zfill(3)
        Image.fromarray(x[i]).save(os.path.join(save_dir, save_name))

def batch_save_images(x, save_dir, save_name, start_idx=0):
    # x is a numpy array of shape BHW or BHW3 or BHW4
    for i in range(x.shape[0]):
        current_save_dir = os.path.join(save_dir, 'sample_%s' % str(i+start_idx).zfill(3))
        os.makedirs(current_save_dir, exist_ok=True)
        Image.fromarray(x[i]).save(os.path.join(current_save_dir, save_name))

def pca_feature_plane(A):
    # A is a tensor of shape BHWC
    B,H,W,C = A.size()
    # pdb.set_trace()
    U,_,_ = torch.pca_lowrank(A.reshape(-1, C), q=3, center=True, niter=2)
    U = U.reshape(B,H,W,3)
    return U

def save_triplane(triplane, save_dir, suffix=None, start_idx=0):
    with torch.no_grad():
        for key in triplane.keys():
            batch_plane = triplane[key] # BCHW
            batch_plane = batch_plane.permute(0,2,3,1) # BHWC
            for k in range(batch_plane.shape[0]):
                plane = batch_plane[k:(k+1)]
                # we want to perform pca for each plane individually
                if plane.shape[-1] > 3:
                    plane = pca_feature_plane(plane)
                plane = (plane - plane.min()) / (plane.max() - plane.min())
                plane = (plane.detach().cpu().numpy()*255).astype(np.uint8)
                batch_save_images(plane, save_dir, 'triplane_feature_%s_plane%s.jpg' % (key, suffix), start_idx=start_idx+k)