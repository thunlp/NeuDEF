# coding=utf8
import torch.storage
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.storage
import torch.nn.functional as F
import torch.nn.init as init

from torch_multi_head_attention import MultiHeadAttention

class attnneudeftb(nn.Module):
    def __init__(self, opt):
        super(attnneudeftb, self).__init__()

        self.vocab_size = opt.vocab_size
        self.term_hidden_size = opt.term_size

        self.use_cuda = opt.cuda

        self.word_emb = nn.Embedding(opt.vocab_size, opt.term_size, padding_idx=0)
        if not (opt.init_emb is None):
            self.word_emb.weight.data.copy_(opt.init_emb.data)

        self.multiheadattn = MultiHeadAttention(in_features=opt.term_size, head_num=opt.n_head)

        # knrm parameters
        tensor_mu = torch.FloatTensor(opt.mu)
        tensor_sigma = torch.FloatTensor(opt.sigma)
        if opt.cuda:
            tensor_mu = tensor_mu.cuda()
            tensor_sigma = tensor_sigma.cuda()
        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, opt.n_bins)
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, opt.n_bins)

        # dense layers
        self.transform_dcq = nn.Linear(self.term_hidden_size, self.term_hidden_size)

        self.qddense = nn.Linear(opt.n_bins, 1, 1)
        self.qbdense = nn.Linear(opt.n_bins, 1, 1)
        self.qcqdense = nn.Linear(opt.n_bins, 1, 1)
        self.dcqdense = nn.Linear(opt.n_bins, 1, 1)
        self.bcqdense = nn.Linear(opt.n_bins, 1, 1)

        self.exp_combine = nn.Linear(2, 1) # (d,cq),(b,cq)
        self.combine = nn.Linear(3, 1) # qd, qe, qb


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
        embed_b = self.word_emb(inputs_dbody)

        qd = F.tanh(self.qddense(self.knrm_logsum(embed_q, embed_d, masks_q, masks_d))) # bs * 1
        qb = F.tanh(self.qbdense(self.knrm_logsum(embed_q, embed_b, masks_q, masks_dbody)))  # bs * 1

        bs, max_num, max_len = inputs_dcq.size()
        concat_inst_dcq = inputs_dcq.view(bs, -1) # bs*(maxnum*len)
        concat_embed_dcq = F.relu(self.transform_dcq(self.word_emb(concat_inst_dcq)))

        concat_encode_dcq = self.multiheadattn(concat_embed_dcq, concat_embed_dcq, concat_embed_dcq) # bs*len*300

        split_embed_dcq = concat_encode_dcq.view(bs, max_num, max_len, -1)
        is_dcq = split_embed_dcq.transpose(0, 1) # max_num * bs * len * 300
        ms_dcq = masks_dcq.transpose(0, 1)

        qcqres = Variable(torch.zeros(max_num, bs, 1))
        qcqres = qcqres.cuda() if self.use_cuda else qcqres

        qcqdattn = Variable(torch.zeros(max_num, bs, 1))
        qcqdattn = qcqdattn.cuda() if self.use_cuda else qcqdattn
        qcqbattn = Variable(torch.zeros(max_num, bs, 1))
        qcqbattn = qcqbattn.cuda() if self.use_cuda else qcqbattn

        encode_d = self.multiheadattn(embed_d, embed_d, embed_d) # bs*len*300
        encode_b = self.multiheadattn(embed_b, embed_b, embed_b) # bs*len*300

        for i in range(max_num):
            i_dcq = is_dcq[i] # bs * len * 300
            m_dcq = ms_dcq[i]
            qcqres[i] = F.tanh(self.qcqdense(self.knrm_logsum(embed_q, i_dcq, masks_q, m_dcq)))  # bs * 1

            qcqdattn[i] = F.tanh(self.dcqdense(self.knrm_logsum(encode_d, i_dcq, masks_d, m_dcq)))  # bs * 1
            qcqbattn[i] = F.tanh(self.bcqdense(self.knrm_logsum(encode_b, i_dcq, masks_dbody, m_dcq)))  # bs * 1

        qcqd = torch.sum(qcqres * qcqdattn, dim=0) # bs*1
        qcqb = torch.sum(qcqres * qcqbattn, dim=0)  # bs*1

        qcq = self.exp_combine(torch.cat([qcqd, qcqb], dim=1))

        sim = F.tanh(self.combine(torch.cat([qd, qb, qcq], dim=1))).squeeze()
        return sim



