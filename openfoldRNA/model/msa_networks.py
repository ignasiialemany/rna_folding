# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

# from model.model.primitives import (
#     Linear, 
#     LayerNorm,
#     Attention, 
#     GlobalAttention, 
#     _attention_chunked_trainable,
# )

#from model.utils.chunk_utils import chunk_layer
#from model.utils.tensor_utils import (
#    permute_final_dims,
#)


class LearnedPositionalEmbedding(nn.Embedding):
    """
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        ''' '''
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class MSANet(nn.Module):
    def __init__(self,
                 d_model = 64,
                 d_msa = 21,
                 padding_idx = None,
                 max_len = 4096,
                 is_pos_emb = True,
                 **unused):
        super(MSANet, self).__init__()
        self.is_pos_emb = is_pos_emb

        self.embed_tokens = nn.Embedding(d_msa, d_model, padding_idx = padding_idx)
        if self.is_pos_emb:
            self.embed_positions = LearnedPositionalEmbedding(max_len, d_model, padding_idx)

    def forward(self, tokens):
        '''

        '''

        B, K, L = tokens.shape
        msa_fea = self.embed_tokens(tokens)

        if self.is_pos_emb:
            msa_fea += self.embed_positions(tokens.reshape(B * K, L)).view(msa_fea.size())

        return msa_fea

    def get_emb_weight(self):
        return self.embed_tokens.weight