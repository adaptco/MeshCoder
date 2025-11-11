# Copyright 2023 The HuggingFace Team. All rights reserved.
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
#
# --------
#
# Modified 2024 by the Tripo AI and Stability AI Team.
#
# Copyright (c) 2024 Tripo AI & Stability AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

import math

# from ...utils import BaseModule
from llama_recipes.custom_models.modules.cross_attn.basic_transformer_block import BasicTransformerBlock

import pdb

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        # p.detach().zero_()
        nn.init.zeros_(p)
    return module

# class Transformer1D(BaseModule):
class SelfAttn(nn.Module):
    def __init__(self,
        # num_query_tokens: int = 128,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        input_group_norm: bool = True,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        gradient_checkpointing: bool = False,
        output_norm: bool=False,
        output_multiplier: float=1,
        zero_init_output: bool=False):

        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.output_multiplier = output_multiplier
        inner_dim = self.num_attention_heads * self.attention_head_dim

        linear_cls = nn.Linear

        # in_channels and out_channels will be set to inner_dim if they are none

        # 1. Define query tokens
        in_channels = inner_dim if in_channels is None else in_channels
        self.in_channels = in_channels
        # self.num_query_tokens = num_query_tokens
        # self.embeddings = nn.Parameter(
        #     torch.randn((in_channels, num_query_tokens), dtype=torch.float32) * 1 / math.sqrt(in_channels)
        # )

        # 2. Define input layers
        if input_group_norm:
            self.norm = torch.nn.GroupNorm(
                num_groups=norm_num_groups,
                num_channels=in_channels,
                eps=1e-6,
                affine=True,
            )
        else:
            self.norm = nn.Identity()
        self.proj_in = linear_cls(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.proj_out = linear_cls(inner_dim, out_channels)

        self.gradient_checkpointing = gradient_checkpointing

        # 5. define skip connections
        self.skip_proj = nn.Identity() if in_channels == out_channels else linear_cls(in_channels, out_channels)

        # 6. define output norm
        self.output_norm = nn.LayerNorm(self.out_channels) if output_norm else nn.Identity()

        self.zero_init_output = zero_init_output
        if zero_init_output:
            if output_norm:
                self.output_norm = zero_module(self.output_norm)
            else:
                self.proj_out = zero_module(self.proj_out)
                self.skip_proj = zero_module(linear_cls(in_channels, out_channels))

    def forward(
        self,
        hidden_states: torch.Tensor,
        # encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        The [`Transformer1DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, sequence len)`:
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.

        Returns:
            torch.FloatTensor
        """
        # hidden_states is of shape B, seq_len, in_channels
        batch = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        # hidden_states = torch.stack([self.embeddings]*batch)
        # hidden_states = self.embeddings.repeat(batch,1,1) # B, in_channels, num_query_tokens
        hidden_states = hidden_states.permute(0, 2, 1).reshape(
            batch, self.in_channels, seq_len
        )

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        # if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        #     encoder_attention_mask = (
        #         1 - encoder_attention_mask.to(hidden_states.dtype)
        #     ) * -10000.0
        #     encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        # residual = hidden_states
        residual = self.skip_proj(hidden_states.permute(0, 2, 1)) # B, num_query_tokens, out_channels

        hidden_states = self.norm(hidden_states)
        # inner_dim = hidden_states.shape[1]
        # hidden_states = hidden_states.permute(0, 2, 1).reshape(
        #     batch, seq_len, inner_dim
        # )
        hidden_states = hidden_states.permute(0, 2, 1).reshape(
            batch, seq_len, self.in_channels
        )
        hidden_states = self.proj_in(hidden_states) # B, num_query_tokens, inner_dim

        # 2. Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    attention_mask,
                    None,
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                )

        # 3. Output
        hidden_states = self.proj_out(hidden_states) # B, num_query_tokens, out_channels
        output = hidden_states + residual
        output = self.output_multiplier * self.output_norm(output)
        return output



if __name__ == '__main__':
    in_channels = 4096
    out_channels = 2048
    cross_attention_dim = None
    num_attention_heads = 32
    attention_head_dim = 128
    # num_query_tokens = 128
    model = SelfAttn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    num_layers=4,
                    cross_attention_dim=cross_attention_dim)
    model.cuda()
    B = 3
    seq_len = 25
    # seq_len2 = 40
    hidden_states = torch.rand(B, seq_len, in_channels).cuda()
    # encoder_hidden_states = torch.rand(B, seq_len2, cross_attention_dim).cuda()
    # out = model(hidden_states, encoder_hidden_states)
    out = model(hidden_states) # shape (B, num_query_tokens, out_channels)
    print(out.shape)
    loss = out.mean()
    loss.backward()
    pdb.set_trace()
