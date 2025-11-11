from llama_recipes.custom_models.modules.pcd_proj.point_projection import PointTriplaneProjector
from llama_recipes.custom_models.modules.dinov2.models.vision_transformer import DinoVisionTransformer
from llama_recipes.custom_models.modules.cross_attn.transformer_1d import QueryCrossAttn

from llama_recipes.custom_models.modules.point_cross_attn.michelangelo_autoencoder import PerceiverCrossAttentionEncoder

import torch
import torch.nn as nn
import time
import pdb

def get_shape_tokenizer(args):
    if 'network_type' in args.keys():
        network_type = args.pop('network_type')
        if network_type == 'triplane_tokenizer':
            model = ShapeTokenizer(**args)
        elif network_type == 'point_cross_attn_tokenizer':
            model = PerceiverCrossAttentionEncoder(**args)
    else:
        model = ShapeTokenizer(**args)
    return model

class ShapeTokenizer(nn.Module):
    def __init__(self, point_projector_args, triplane_tokenizer_args, querier_args, offset_tensor=None):
        super().__init__()

        plane_resolution = point_projector_args['plane_resolution']
        triplane_tokenizer_args['img_size'] = (plane_resolution, plane_resolution*3)
        triplane_tokenizer_args['in_chans'] = point_projector_args['plane_fea_dim']
        querier_args['cross_attention_dim'] = triplane_tokenizer_args['embed_dim']

        self.point_projector = PointTriplaneProjector(**point_projector_args)
        self.triplane_tokenizer = DinoVisionTransformer(**triplane_tokenizer_args)
        if querier_args.get('apply_query', True):
            self.querier = QueryCrossAttn(**querier_args)
        else:
            if querier_args['output_norm']:
                self.querier = [nn.Linear(triplane_tokenizer_args['embed_dim'], querier_args['out_channels']), 
                                nn.LayerNorm(querier_args['out_channels'])]
                self.querier = nn.Sequential(*self.querier)
            else:
                self.querier = nn.Linear(triplane_tokenizer_args['embed_dim'], querier_args['out_channels'])
        self.apply_offset = not offset_tensor is None
        if self.apply_offset:
            assert offset_tensor.shape[0] == self.querier.out_channels
            self.register_buffer('offset', offset_tensor.unsqueeze(0).unsqueeze(0))
            # of shape (1, 1, out_channels)
    
    def forward(self, x, mask=None, verbose=False):
        # x is a point cloud of shape B,N,point_fea_dim
        # mask is a tensor of shape B,N that contain 0 and 1
        # 1 indicates valid point and 0 indicate padded point
        # we assume its first 3 dimensions are 3D coodinates, next 3 dimensions are normals, other dimensions are other features
        plane_features = self.point_projector(x, point_mask=mask)
        triplane_feature = torch.cat([plane_features['xy'], plane_features['yz'], plane_features['zx']], dim=3)
        # triplane_feature is of shape B, plane_fea_dim, plane_resolution, 3*plane_resolution
        shape_tokens = self.triplane_tokenizer(triplane_feature, is_training=True)['x_norm_patchtokens']
        # shape_tokens is of shape B, plane_resolution/patch_size * 3*plane_resolution/patch_size, embed_dim
        # shape_tokens is of shape B, num_register_tokens + num_cls_tokens + plane_resolution/patch_size * 3*plane_resolution/patch_size, embed_dim
        queried_tokens = self.querier(shape_tokens)
        # queried_tokens is of shape (B, num_query_tokens, out_channels)
        if self.apply_offset:
            queried_tokens = queried_tokens + self.offset
        if verbose:
            print('triplane_feature', triplane_feature.shape)
            print('shape_tokens', shape_tokens.shape)
            print('queried_tokens', queried_tokens.shape)
        return queried_tokens

        

if __name__ == '__main__':
    point_fea_dim = 6
    plane_fea_dim = 32
    plane_resolution = 128
    embed_dim = 768
    object_scale = 1
    point_projector_args = {
        'plane_fea_dim': plane_fea_dim, 
        'point_fea_dim': point_fea_dim, 
        'hidden_dim': 128, 
        'scatter_type': 'max', 
        'norm': 'Identity',
        'plane_resolution': plane_resolution, 
        'plane_type': ['xy', 'yz', 'zx'], 
        'n_blocks': 3,
        'object_scale': object_scale, 
        'plane_scale': 1.05, 
        'pcd_clamp_scale': 1,
        'use_pool_local': True,
        'use_pos_emb': True,
        'pos_emb_multires': 10}
    triplane_tokenizer_args = {
        # 'img_size': (plane_resolution,plane_resolution*3),
        'patch_size': 16,
        # 'in_chans': plane_fea_dim,
        'embed_dim': embed_dim,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'qkv_bias': True,
        'ffn_bias': True,
        'proj_bias': True,
        'drop_path_rate': 0.0,
        'drop_path_uniform': False,
        'init_values': None,  # for layerscale: None or 0 : > no layerscale
        'ffn_layer': "mlp",
        'block_chunks': 1,
        'num_register_tokens': 0,
        'interpolate_antialias': False,
        'interpolate_offset': 0.1}
    querier_args = {
        'num_query_tokens': 128, 
        'in_channels': None,
        'out_channels': 4096,
        'num_attention_heads': 16,
        'attention_head_dim': 64,
        'num_layers': 12,
        # 'cross_attention_dim': embed_dim
        }
    shape_tokenizer = ShapeTokenizer(point_projector_args, triplane_tokenizer_args, querier_args)
    shape_tokenizer.cuda()
    points = (torch.rand(32, 16384, point_fea_dim).cuda() - 0.5) * 2 * object_scale
    start = time.time()
    tokens = shape_tokenizer(points, verbose=True)
    print('forward time %.2f' % (time.time()-start))
    print(tokens.shape)
    pdb.set_trace()