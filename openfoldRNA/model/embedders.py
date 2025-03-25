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
from typing import Tuple

#from model.model.primitives import Linear, LayerNorm
#from model.utils.tensor_utils import add
import model.rna_fm as rna_esm
from model.msa_networks import MSANet, PairNet
#from model.pair import PairNet
from model.utils.alphabet import RNAAlphabet
import os

class MSAEmbedder(nn.Module):
    """MSAEmbedder """

    def __init__(self,
                 c_m,
                 c_z,
                 rna_fm=None,
                 ):
        super().__init__()

        self.rna_fm, self.rna_fm_reduction = None, None
        self.mask_rna_fm_tokens = False

        self.alphabet = RNAAlphabet.from_architecture('RNA')

        self.msa_emb = MSANet(d_model = c_m,
                               d_msa = len(self.alphabet),
                               padding_idx = self.alphabet.padding_idx,
                               is_pos_emb = True,
                               )

        self.pair_emb = PairNet(d_model = c_z,
                                 d_msa = len(self.alphabet),
                                 )

        self.rna_fm, self.rna_fm_reduction = None, None

        if exists(rna_fm) and rna_fm['enable']:
            # Load RNA-FM model
            self.rna_fm_dim = 640
            self.rna_fm, _ = rna_esm.pretrained.esm1b_rna_t12()
            self.rna_fm.eval()
            for param in self.rna_fm.parameters():
                param.detach_()
            self.rna_fm_reduction = nn.Linear(self.rna_fm_dim + c_m, c_m)

            # rna_fm_ckpt = './rna_fm.pt'
            # if not os.path.exists(rna_fm_ckpt):
            #     torch.save({'model': self.rna_fm.state_dict()}, rna_fm_ckpt)

    def forward(self, tokens, rna_fm_tokens = None, is_BKL = True, **unused):

        assert tokens.ndim == 3
        if not is_BKL:
            tokens = tokens.permute(0, 2, 1)

        B, K, L = tokens.size()# batch_size, num_alignments, seq_len
        msa_fea = self.msa_emb(tokens)

        if exists(self.rna_fm):
            results = self.rna_fm(rna_fm_tokens, need_head_weights=False, repr_layers=[12], return_contacts=False)
            token_representations = results["representations"][12].unsqueeze(1).expand(-1, K, -1, -1)
            msa_fea = self.rna_fm_reduction(torch.cat([token_representations, msa_fea], dim = -1))

        pair_fea = self.pair_emb(tokens, t1ds = None, t2ds = None)

        return msa_fea, pair_fea