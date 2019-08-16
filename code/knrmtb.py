# coding=utf8
import torch.storage
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.storage
import torch.nn.functional as F
import torch.nn.init as init

class knrmtb(nn.Module):
    def __init__(self, opt):
        super(knrmtb, self).__init__()

        self.vocab_size = opt.vocab_size
        self.term_hidden_size = opt.term_size

        self.use_cuda = opt.cuda

        # knrm parameters
        tensor_mu = torch.FloatTensor(opt.mu)
        tensor_sigma = torch.FloatTensor(opt.sigma)
        if opt.cuda:
            tensor_mu = tensor_mu.cuda()
            tensor_sigma = tensor_sigma.cuda()
        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, opt.n_bins)
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, opt.n_bins)

        self.word_emb = nn.Embedding(opt.vocab_size, opt.term_size, padding_idx=0)
        if not (opt.init_emb is None):
            self.word_emb.weight.data.copy_(opt.init_emb.data)

        self.qddense = nn.Linear(opt.n_bins, 1, 1)
        self.qbdense = nn.Linear(opt.n_bins, 1, 1)

        self.combine = nn.Linear(2, 1) # qd, qb

    def knrm_logsum(self, embed_q, embed_d, masks_q, masks_d):
        attn_d = masks_d.view(masks_d.size()[0], 1, masks_d.size()[1], 1)
        attn_q = masks_q.view(masks_q.size()[0], masks_q.size()[1], 1)
        q_norm = F.normalize(embed_q, 2, 2)
        d_norm = F.normalize(embed_d, 2, 2)
        sim = torch.bmm(q_norm, torch.transpose(d_norm, 1, 2)) # bs*lenq*lend, cos sim
        sim = sim.view(q_norm.size()[0], q_norm.size()[1],d_norm.size()[1], 1)
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) # bs*lenq*lend*n_bins
        pooling_value = pooling_value * attn_d
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * attn_q * 0.01
        log_pooling_sum = torch.sum(log_pooling_sum, 1)

        return log_pooling_sum

    def forward(self, inputs_q, inputs_d, inputs_dcq, inputs_dbody, masks_q, masks_d, masks_dcq, masks_dbody):
        embed_q = self.word_emb(inputs_q)
        embed_d = self.word_emb(inputs_d)
        embed_db = self.word_emb(inputs_dbody)

        qd = F.tanh(self.qddense(self.knrm_logsum(embed_q, embed_d, masks_q, masks_d))) # bs * 1
        qb = F.tanh(self.qbdense(self.knrm_logsum(embed_q, embed_db, masks_q, masks_dbody))) # bs * 1

        sim = self.combine(torch.cat([qd, qb], dim=1))
        return sim.squeeze()



