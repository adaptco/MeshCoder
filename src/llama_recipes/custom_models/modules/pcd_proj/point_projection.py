import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_mean, scatter_max
# from triplane_autoencoder.unet import UNet
# from triplane_autoencoder.resnet_block import ResnetBlockFC
from llama_recipes.custom_models.modules.pcd_proj.resnet_block import ResnetBlockFC
from llama_recipes.custom_models.modules.pcd_proj.util import get_embedder, save_triplane
# from triplane_autoencoder.guided_diffusion.unet import UNetModel, ResBlock
# from triplane_autoencoder.util import save_triplane, get_embedder
# from triplane_autoencoder.guided_diffusion.nn import zero_module

# from pointnet2.models.model_utils import get_embedder

import torch.nn.init as init
import numpy as np
import copy
import os

import pdb

def normalize_pcd(x, desired_scale=1):
    # x is a tensor of shape BN3
    # we will normalize it and make it range from -desired_scale to desired_scale
    maxx = x.max(dim=1, keepdims=True)[0] # B,1,3
    minn = x.min(dim=1, keepdims=True)[0] # B,1,3
    center = (maxx + minn)/ 2 # B,1,3
    max_length = (maxx - minn).max(dim=2, keepdims=True)[0] # B,1,1
    x = x -center
    x = x / max_length * 2 * desired_scale
    return x

class PointTriplaneProjector(nn.Module):
    def __init__(self, plane_fea_dim=128, point_fea_dim=3, hidden_dim=128, scatter_type='max', 
                 plane_resolution=None, plane_type=['xy', 'yz', 'zx'], n_blocks=5, norm='Identity',
                 object_scale=1, plane_scale=1.15, 
                 pcd_clamp_scale=None,
                 use_pool_local=True,
                 use_pos_emb=False,
                 pos_emb_multires=10,
                 in_proj=False):
        super().__init__()

        # plane_fea_dim (int): plane_fea_dim is the dimension of the features at the triplane, dimension of latent code c
        # point_fea_dim (int): input points dimension
        # hidden_dim (int): dimension of the fc layers in pool local operation
        # scatter_type (str): feature aggregation when doing local pooling
        # plane_resolution (int): defined resolution for plane feature
        # plane_type (str): feature type, 'zx' - 1-plane, ['zx', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        # n_blocks (int): number of blocks of ResNetBlockFC layers in pool local operation

        self.object_scale = object_scale # we normalize input point cloud to [-object_scale, object_scale]^3 
        # if explicitly_normalize_input_pcd == True in forward
        self.plane_scale = plane_scale # we assume the triplane spans the cube [-plane_scale, plane_scale]^3
        # we clamp input pcd to [-pcd_clamp_scale,pcd_clamp_scale]^3 before project it to the 
        # pcd_clamp_scale should be slightly less than plane_scale to ensure stable training
        if pcd_clamp_scale is None:
            self.pcd_clamp_scale = plane_scale
        else:
            self.pcd_clamp_scale = pcd_clamp_scale
        
        self.plane_resolution = plane_resolution
        self.plane_type = plane_type

        # if scatter_type == 'max':
        #     self.scatter = scatter_max
        # elif scatter_type == 'mean':
        #     self.scatter = scatter_mean
        self.scatter_type = scatter_type

        # we need to make sure that pcd do not go outside the plane
        assert (self.pcd_clamp_scale/2/self.plane_scale + 0.5) * self.plane_resolution <= (self.plane_resolution - 1)

        self.plane_fea_dim = plane_fea_dim # point_fea_dim is the dimension of the input points, including 3D coordinates

        norm_dict = {'Identity': nn.Identity, 'LayerNorm': nn.LayerNorm}
        norm_layer = norm_dict[norm]

        # setup position encoding for input point cloud
        self.use_pos_emb = use_pos_emb
        if self.use_pos_emb:
            self.pos_encode, pos_encode_dim = get_embedder(pos_emb_multires)
            self.pos_encode_proj = torch.nn.Sequential(
                    nn.Linear(pos_encode_dim, pos_encode_dim),
                    norm_layer(pos_encode_dim),
                    nn.GELU(),
                    nn.Linear(pos_encode_dim, pos_encode_dim)
                )
            point_fea_dim = point_fea_dim + pos_encode_dim
        
        self.in_proj = in_proj
        if self.in_proj:
            self.input_projection = nn.Linear(point_fea_dim, hidden_dim)
            resblock_in_dim = hidden_dim
        else:
            resblock_in_dim = point_fea_dim

        # setup layers to transform input point cloud coordinates and features before projecting them to triplane
        self.use_pool_local = use_pool_local
        if self.use_pool_local:
            assert n_blocks >= 3
            layer_list =  ([ResnetBlockFC(resblock_in_dim, hidden_dim)] +
                                [ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks-2)] + 
                                [ResnetBlockFC(2*hidden_dim, plane_fea_dim)])
            self.blocks = nn.ModuleList(layer_list)
        else:
            assert n_blocks >= 1
            if n_blocks == 1:
                self.blocks = ResnetBlockFC(resblock_in_dim, plane_fea_dim)
            else:
                layer_list =  ([ResnetBlockFC(resblock_in_dim, hidden_dim)] +
                                [ResnetBlockFC(hidden_dim, hidden_dim) for i in range(n_blocks-2)] + 
                                [ResnetBlockFC(hidden_dim, plane_fea_dim)])
                self.blocks = nn.Sequential(*layer_list)

    def forward(self, p, point_mask=None, visualize_triplane=False, triplane_save_dir=None, start_idx=0, 
                explicitly_normalize_input_pcd=False):
        # p is of shape B,N,point_fea_dim, we assume its first 3 dimensions are 3D coordinates
        # point_mask is of shape B,N and contain 1 and 0, it indicates valid points (1) and padded points (0)
        # explicitly_normalize_input_pcd: whether mannully normalize the input pcd, 
        # this is used when the input point cloud is not normalized
        
        batch_size, N, D = p.size()

        if explicitly_normalize_input_pcd:
            xyz = p[:,:,0:3]
            xyz = normalize_pcd(xyz, desired_scale=self.object_scale)
            p = torch.cat([xyz, p[:,:,3:]], dim=2)
        
        if not point_mask is None:
            device = p.device
            point_mask = point_mask.float().unsqueeze(-1) # B,N,1
            p = p * point_mask # we set padding points with zero position and zero features
            corner_position = (self.object_scale + self.plane_scale) / 2
            xyz = p[:,:,0:3] * point_mask + torch.ones(batch_size, N, 3).to(device) * corner_position * (1-point_mask)
            # we move positions of the padding points to the corner so that they donot affect other points
            p = torch.cat([xyz, p[:,:,3:]], dim=2)

        if self.use_pos_emb:
            pos_emb = self.pos_encode_proj(self.pos_encode(p[:,:,0:3]))
            p = torch.cat([p, pos_emb], dim=2)
        
        if self.use_pool_local:
            # acquire the index for each point
            coord = {}
            index = {}
            for key in self.plane_type:
                coord[key] = self.normalize_coordinate(p, plane=key, plane_scale=self.plane_scale)
                index[key] = self.coordinate2index(coord[key], self.plane_resolution)

            if self.in_proj:
                net = self.input_projection(p)
                net = self.blocks[0](net)
            else:
                net = self.blocks[0](p)
            for block in self.blocks[1:]:
                pooled = self.pool_local(index, net)
                net = torch.cat([net, pooled], dim=2)
                net = block(net)
            # c = self.fc_c(net)
            c = net
        else:
            if self.in_proj:
                c = self.blocks(self.input_projection(p))
            else:
                c = self.blocks(p)

        fea = {}
        for key in self.plane_type:
            fea[key] = self.generate_plane_features(p, c, plane=key)
        
        # pdb.set_trace()
        # visualize_triplane = True
        # triplane_save_dir = 'triplane_vis'
        if visualize_triplane:
            save_triplane(fea, triplane_save_dir, suffix='_before_unet', start_idx=start_idx)
            xyz_normal = p[:,:,0:6]
            for k in range(xyz_normal.shape[0]):
                np.savetxt(os.path.join(triplane_save_dir, 'sample_%s' % str(k+start_idx).zfill(3), 'pcd.xyz'), 
                        xyz_normal[k].detach().cpu().numpy())
            pdb.set_trace()
        return fea


    def normalize_coordinate(self, p, plane_scale=1.15, plane='zx'):
        # p is the position tensor of shape BN3
        # the order of the axis is very important for 3D aware convolution, we use xy, yz, and zx 
        try:
            if plane == 'zx':
                xy = p[:, :, [2, 0]]
            elif plane == 'xy':
                xy = p[:, :, [0, 1]]
            elif plane == 'yz':
                xy = p[:, :, [1, 2]]
            else:
                raise Exception('plane %s is not supported' % plane)
        except:
            print(p.device, p.shape, p.dtype)

        # xy_new = torch.clamp(xy, min=-plane_scale+1e-6, max=plane_scale-1e-6)
        xy_new = torch.clamp(xy, min=-self.pcd_clamp_scale, max=self.pcd_clamp_scale)
        # plane original is considered in -plane_scale, plane_scale, now it is normalized to 0,1, 
        # and discretized to 0~plane_resolution
        # plane coord = plane coord / 2 / plane_scale + 0.5, range to 0 and 1 now
        xy_new = xy_new / plane_scale / 2 + 0.5 # range to 0 and 1 now
        return xy_new


    def coordinate2index(self, x, reso):
        # x is of shape BN2
        x = (x * reso).long()
        index = x[:, :, 0] + reso * x[:, :, 1] # BN
        index = index[:, None, :] # B,1,N
        return index


    # xy is the normalized coordinates of the point cloud of each plane 
    # I'm pretty sure the keys of xy are the same as those of index, so xy isn't needed here as input 
    def pool_local(self, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        # keys = xy.keys()
        keys = index.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            fea = scatter(c.permute(0, 2, 1), index[key], dim_size=self.plane_resolution**2, reduce=self.scatter_type)
            # if self.scatter == scatter_max:
            #     fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)


    def generate_plane_features(self, p, c, plane='zx'):
        # acquire indices of features in plane
        xy = self.normalize_coordinate(p, plane=plane, plane_scale=self.plane_scale)
        index = self.coordinate2index(xy, self.plane_resolution)

        # scatter plane features from points
        # fea_plane = c.new_zeros(p.size(0), self.plane_fea_dim, self.plane_resolution**2)
        c = c.permute(0, 2, 1) # BCN
        # pdb.set_trace()
        # print(p.max().item(), p.min().item(), index.min().item(), self.plane_resolution**2-1-index.max().item(), 
        #     c.min().item(), c.max().item())
        # index = torch.clamp(index, min=0, max=self.plane_resolution**2-1)
        # fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        # fea_plane = self.scatter(c, index, out=fea_plane)
        # fea_plane = self.scatter(c, index, dim_size=self.plane_resolution**2)
        fea_plane = scatter(c, index, dim_size=self.plane_resolution**2, reduce=self.scatter_type)
        # pdb.set_trace()
        fea_plane = fea_plane.reshape(p.size(0), self.plane_fea_dim, self.plane_resolution, self.plane_resolution) # sparse matrix (B x 512 x reso x reso)
        # for those plane pixels that do not correspond to any points, its feature value will be 0 by default 

        return fea_plane

def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)

if __name__ == '__main__':
    B = 4
    N = 20
    point_fea_dim = 6
    object_scale = 1
    plane_scale = 1.15
    dmtet_scale = 1.1
    feature_add_mode = 'concat'
    plane_fea_dim = 32
    plane_type = ['xy', 'yz', 'zx']
    plane_resolution = 256
    use_3D_aware_conv = True

    triplane_decoder = PointTriplaneProjector(plane_fea_dim=plane_fea_dim, point_fea_dim=point_fea_dim, 
                hidden_dim=128, scatter_type='max', norm='Identity',
                 plane_resolution=plane_resolution, plane_type=['xy', 'yz', 'zx'], n_blocks=3,
                 object_scale=1, plane_scale=1.05, 
                 pcd_clamp_scale=1,
                 use_pool_local=True,
                 use_pos_emb=True,
                 pos_emb_multires=10)
    
    points = (torch.rand(B,N,point_fea_dim)-0.5) * 2 * object_scale
    fea = triplane_decoder(points)
    for key in fea.keys():
        print(key, fea[key].shape)
    print(triplane_decoder.blocks)
    pdb.set_trace()
    


    
