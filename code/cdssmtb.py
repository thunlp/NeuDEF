import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np

class cdssmtb(nn.Module):
    def __init__(self, opt):
        super(cdssmtb, self).__init__()
        self.vocab_size = opt.vocab_size
        self.d_word_vec = opt.term_size
        self.use_cuda = opt.cuda

        self.word_emb = nn.Embedding(opt.vocab_size, opt.term_size, padding_idx=0)
        if opt.init_emb is not None:
            self.word_emb.weight.data.copy_(opt.init_emb.data)

        self.dense_f = nn.Linear(opt.n_bins * 9, 1, 1)
        self.tanh = nn.Tanh()

        self.conv_tri = nn.Sequential(
            nn.Conv2d(1, self.d_word_vec, (3, self.d_word_vec)),
            nn.ReLU()
        )

        self.trans_q = nn.Linear(self.d_word_vec, self.d_word_vec)
        self.trans_d = nn.Linear(self.d_word_vec, self.d_word_vec)
        self.trans_b = nn.Linear(self.d_word_vec, self.d_word_vec)

        self.combine = nn.Linear(2, 1)

    def forward(self, inputs_q, inputs_d, inputs_dcq, inputs_dbody, masks_q, masks_d, masks_dcq, masks_dbody):
        bs, lenq = inputs_q.size()
        bs, lend = inputs_d.size()
        bs, lendb = inputs_dbody.size()

        q_embed = self.word_emb(inputs_q) # bs*10*300
        d_embed = self.word_emb(inputs_d) # bs*50*300
        b_embed = self.word_emb(inputs_dbody) # bs*100*300

        qt_conv = self.conv_tri(q_embed.view(inputs_q.size()[0], 1, -1, self.d_word_vec)) # bs*300*(lenq-2)*1
        dt_conv = self.conv_tri(d_embed.view(inputs_d.size()[0], 1, -1, self.d_word_vec)) # bs*300*(lend-2)*1
        bt_conv = self.conv_tri(b_embed.view(inputs_dbody.size()[0], 1, -1, self.d_word_vec))  # bs*300*(lendb-2)*1

        qt_maxpool = F.max_pool2d(qt_conv, (lenq - 2, 1), stride=(1, 1), padding=0).squeeze().unsqueeze(1) # bs*1*300
        dt_maxpool = F.max_pool2d(dt_conv, (lend - 2, 1), stride=(1, 1), padding=0).squeeze().unsqueeze(1) # bs*1*300
        bt_maxpool = F.max_pool2d(bt_conv, (lendb - 2, 1), stride=(1, 1), padding=0).squeeze().unsqueeze(1) # bs*1*300

        q_seman = self.trans_q(qt_maxpool)
        d_seman = self.trans_d(dt_maxpool)
        b_seman = self.trans_b(bt_maxpool)

        cos_qd = F.cosine_similarity(q_seman, d_seman, dim=2) # bs*1
        cos_qb = F.cosine_similarity(q_seman, b_seman, dim=2) # bs*1

        sim = self.combine(torch.cat([cos_qd, cos_qb], dim=1))
        return sim.squeeze()
