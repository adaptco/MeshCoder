from llama_recipes.custom_models.modules.dinov2.models.vision_transformer import DinoVisionTransformer

import torch

import pdb

if __name__ == '__main__':
    model = DinoVisionTransformer(img_size=(256,768),
        patch_size=8,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        # embed_layer=PatchEmbed,
        # act_layer=nn.GELU,
        # block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    model.cuda()
    x = torch.rand(5, 3, 256, 768).cuda()
    out = model(x, is_training=True)
    print(out['x_prenorm'].shape)
    pdb.set_trace()