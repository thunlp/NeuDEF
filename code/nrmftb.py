# coding=utf8
import torch.storage
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.storage
import torch.nn.functional as F
import torch.nn.init as init

class nrmftb(nn.Module):
    def __init__(self, opt):
        super(nrmftb, self).__init__()

        self.vocab_size = opt.vocab_size
        self.hidden = 64
        self.term_hidden_size = opt.term_size

        self.use_cuda = opt.cuda

        self.word_emb = nn.Embedding(opt.vocab_size, opt.term_size, padding_idx=0)
        if not (opt.init_emb is None):
            self.word_emb.weight.data.copy_(opt.init_emb.data)
    
        # conv
        self.conv_tri = nn.Sequential(
            nn.Conv2d(1, self.hidden, (3, self.term_hidden_size)),
            nn.ReLU()
        )
        self.trans_q = nn.Linear(self.hidden, self.hidden * 3)
        self.trans_d = nn.Linear(self.hidden, self.hidden)
        self.trans_b = nn.Linear(self.hidden, self.hidden)
        self.trans_cq = nn.Linear(self.hidden, self.hidden)

        self.match_dense = nn.Linear(self.hidden *3, 1)

    def cdssm_repre(self, embed):
        bs, maxlen, termsize = embed.size()
        t_conv = self.conv_tri(embed.view(bs, 1, -1, self.term_hidden_size)) # bs*300*(lenq-2)*1
        t_maxpool = F.max_pool2d(t_conv, (maxlen - 2, 1), stride=(1, 1), padding=0).squeeze().unsqueeze(1) # bs*1*300
        return t_maxpool
        
    def forward(self, inputs_q, inputs_d, inputs_dcq, inputs_dbody, masks_q, masks_d, masks_dcq, masks_dbody):
        embed_q = self.word_emb(inputs_q)
        embed_d = self.word_emb(inputs_d)
        embed_b = self.word_emb(inputs_dbody)
        q_repre = self.trans_q(self.cdssm_repre(embed_q)) # bs*1*(64*3)

        dtitle_repre = self.trans_d(self.cdssm_repre(embed_d)) # bs*1*64
        dbody_repre = self.trans_b(self.cdssm_repre(embed_b)) # bs*1*64

        bs, max_num, max_len = inputs_dcq.size()
        concat_inst_dcq = inputs_dcq.view(bs, -1) # bs*(maxnum*len)
        split_embed_dcq = self.word_emb(concat_inst_dcq).view(bs, max_num, max_len, -1)

        is_dcq = split_embed_dcq.transpose(0, 1) # max_num * bs * len * 300
        ms_dcq = masks_dcq.transpose(0, 1)

        cqs = Variable(torch.zeros(max_num, bs, 1, self.hidden))
        cqs = cqs.cuda() if self.use_cuda else cqs

        for i in range(max_num):
            i_dcq = is_dcq[i] # bs * len * 300
            m_dcq = ms_dcq[i]
            cq_repre = self.trans_cq(self.cdssm_repre(i_dcq)) # bs*1*64
            cqs[i] = cq_repre

        cq_repre = torch.mean(cqs, dim=0) # bs*1*64

        d_repre = torch.cat([dtitle_repre, dbody_repre, cq_repre], dim=2) # bs*1*(64*3)

        qd = q_repre * d_repre
        sim = self.match_dense(qd).squeeze().unsqueeze(1) #bs*1

        return sim.squeeze()



