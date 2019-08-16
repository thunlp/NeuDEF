import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np

class cknrmtb(nn.Module):
    def __init__(self, opt):
        super(cknrmtb, self).__init__()
        self.vocab_size = opt.vocab_size
        self.d_word_vec = opt.term_size
        self.use_cuda = opt.cuda

        tensor_mu = torch.FloatTensor(opt.mu)
        tensor_sigma = torch.FloatTensor(opt.sigma)
        if opt.cuda:
            tensor_mu = tensor_mu.cuda()
            tensor_sigma = tensor_sigma.cuda()
        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, opt.n_bins)
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, opt.n_bins)

        self.word_emb = nn.Embedding(opt.vocab_size, opt.term_size, padding_idx=0)
        if opt.init_emb is not None:
            self.word_emb.weight.data.copy_(opt.init_emb.data)

        self.tanh = nn.Tanh()

        self.conv_uni = nn.Sequential(
            nn.Conv2d(1, 128, (1, self.d_word_vec)),
            nn.ReLU()
        )
        self.conv_bi = nn.Sequential(
            nn.Conv2d(1, 128, (2, self.d_word_vec)),
            nn.ReLU()
        )
        self.conv_tri = nn.Sequential(
            nn.Conv2d(1, 128, (3, self.d_word_vec)),
            nn.ReLU()
        )

        self.qddense = nn.Linear(opt.n_bins * 9, 1, 1)
        self.qbdense = nn.Linear(opt.n_bins * 9, 1, 1)

        self.combine = nn.Linear(2, 1) # qd, qb

    def get_intersect_matrix(self, q_embed, d_embed, atten_q, atten_d):
        sim = torch.bmm(q_embed, d_embed).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[2], 1)
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * atten_d
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01 * atten_q
        log_pooling_sum = torch.sum(log_pooling_sum, 1)
        return log_pooling_sum

    def convknrm_logsum(self, embed_q, embed_d, masks_q, masks_d):
        bs, qlen, dim = embed_q.size()
        bs, dlen, dim = embed_d.size()

        qu_embed = torch.transpose(torch.squeeze(self.conv_uni(embed_q.view(bs, 1, -1, self.d_word_vec))), 1, 2) + 0.000000001
        qb_embed = torch.transpose(torch.squeeze(self.conv_bi(embed_q.view(bs, 1, -1, self.d_word_vec))), 1, 2) + 0.000000001
        qt_embed = torch.transpose(torch.squeeze(self.conv_tri(embed_q.view(bs, 1, -1, self.d_word_vec))), 1, 2) + 0.000000001
        du_embed = torch.squeeze(self.conv_uni(embed_d.view(bs, 1, -1, self.d_word_vec))) + 0.000000001
        db_embed = torch.squeeze(self.conv_bi(embed_d.view(bs, 1, -1, self.d_word_vec))) + 0.000000001
        dt_embed = torch.squeeze(self.conv_tri(embed_d.view(bs, 1, -1, self.d_word_vec))) + 0.000000001
        qu_embed_norm = F.normalize(qu_embed, p=2, dim=2, eps=1e-10)
        qb_embed_norm = F.normalize(qb_embed, p=2, dim=2, eps=1e-10)
        qt_embed_norm = F.normalize(qt_embed, p=2, dim=2, eps=1e-10)
        du_embed_norm = F.normalize(du_embed, p=2, dim=1, eps=1e-10)
        db_embed_norm = F.normalize(db_embed, p=2, dim=1, eps=1e-10)
        dt_embed_norm = F.normalize(dt_embed, p=2, dim=1, eps=1e-10)

        mask_q = masks_q.view(masks_q.size()[0], masks_q.size()[1], 1)
        mask_d = masks_d.view(masks_d.size()[0], 1, masks_d.size()[1], 1)
        mask_qu = mask_q[:, :qlen - (1 - 1), :]
        mask_qb = mask_q[:, :qlen - (2 - 1), :]
        mask_qt = mask_q[:, :qlen - (3 - 1), :]
        mask_du = mask_d[:, :, :dlen - (1 - 1), :]
        mask_db = mask_d[:, :, :dlen - (2 - 1), :]
        mask_dt = mask_d[:, :, :dlen - (3 - 1), :]

        log_pooling_sum_wwuu = self.get_intersect_matrix(qu_embed_norm, du_embed_norm, mask_qu, mask_du)
        log_pooling_sum_wwub = self.get_intersect_matrix(qu_embed_norm, db_embed_norm, mask_qu, mask_db)
        log_pooling_sum_wwut = self.get_intersect_matrix(qu_embed_norm, dt_embed_norm, mask_qu, mask_dt)
        log_pooling_sum_wwbu = self.get_intersect_matrix(qb_embed_norm, du_embed_norm, mask_qb, mask_du)
        log_pooling_sum_wwbb = self.get_intersect_matrix(qb_embed_norm, db_embed_norm, mask_qb, mask_db)
        log_pooling_sum_wwbt = self.get_intersect_matrix(qb_embed_norm, dt_embed_norm, mask_qb, mask_dt)
        log_pooling_sum_wwtu = self.get_intersect_matrix(qt_embed_norm, du_embed_norm, mask_qt, mask_du)
        log_pooling_sum_wwtb = self.get_intersect_matrix(qt_embed_norm, db_embed_norm, mask_qt, mask_db)
        log_pooling_sum_wwtt = self.get_intersect_matrix(qt_embed_norm, dt_embed_norm, mask_qt, mask_dt)

        log_pooling_sum = torch.cat(
            [log_pooling_sum_wwuu, log_pooling_sum_wwub, log_pooling_sum_wwut, \
             log_pooling_sum_wwbu, log_pooling_sum_wwbb, log_pooling_sum_wwbt, \
             log_pooling_sum_wwtu, log_pooling_sum_wwtb, log_pooling_sum_wwtt], 1)

        return log_pooling_sum


    def forward(self, inputs_q, inputs_d, inputs_dcq, inputs_dbody, masks_q, masks_d, masks_dcq, masks_dbody):
        embed_q = self.word_emb(inputs_q)
        embed_d = self.word_emb(inputs_d)
        embed_db = self.word_emb(inputs_dbody)

        qd = F.tanh(self.qddense(self.convknrm_logsum(embed_q, embed_d, masks_q, masks_d))) # bs * 1
        qb = F.tanh(self.qbdense(self.convknrm_logsum(embed_q, embed_db, masks_q, masks_dbody))) # bs * 1

        sim = self.combine(torch.cat([qd, qb], dim=1))
        return sim.squeeze()

