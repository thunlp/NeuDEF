''' Data Loader class for test iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable
from config import MAX_Q_LEN, MAX_D_LEN, MAX_EXP_LEN, MAX_BODY_LEN

class DataLoaderTest(object):
    def __init__(self, inputs_q=None, inputs_d=None, inputs_qid=None, inputs_dsogouid=None,\
                 inputs_dcq=None, inputs_dbody=None,\
                 cuda=True, batch_size=1, shuffle=True, test=False):
        assert inputs_q
        assert inputs_d
        assert inputs_qid
        assert inputs_dsogouid
        assert inputs_dcq
        assert inputs_dbody

        assert len(inputs_q)==len(inputs_d)==len(inputs_qid)==len(inputs_dsogouid)==len(inputs_dcq)==len(inputs_dbody)

        self._query_num = len(inputs_q)

        self._n_batch = int(np.ceil(len(inputs_q) / batch_size))
        self._batch_size = batch_size

        self.cuda = cuda
        self.test = test

        self._inputs_q = inputs_q
        self._inputs_d = inputs_d
        self._inputs_qid = inputs_qid
        self._inputs_dsogouid = inputs_dsogouid
        self._inputs_dcq = inputs_dcq
        self._inputs_dbody = inputs_dbody

        self._iter_count = 0

        self._need_shuffle = shuffle
        if self._need_shuffle:
            self.shuffle()

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._inputs_q)

    def shuffle(self):
        ''' Shuffle data for a brand new start '''
        paired_insts = list(zip(self._inputs_q, self._inputs_d, self._inputs_qid, self._inputs_dsogouid,\
                                self._inputs_dcq, self._inputs_dbody))
        random.shuffle(paired_insts)
        self._inputs_q, self._inputs_d, self._inputs_qid, self._inputs_dsogouid,\
        self._inputs_dcq, self._inputs_dbody = zip(*paired_insts)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next sample '''

        def pad_to_longest(insts, max_len):
            ''' Pad the instance to the max seq length in batch '''
            inst_data = np.array([
                inst[:max_len] + [0] * (max_len - len(inst[:max_len]))
                for inst in insts])
            mask = np.zeros((inst_data.shape[0], inst_data.shape[1]))
            for b in range(len(inst_data)):
                for i in range(len(inst_data[b])):
                    if inst_data[b, i] > 0:
                        mask[b, i] = 1
            mask_tensor = Variable(
                    torch.FloatTensor(mask), requires_grad = False)
            inst_data_tensor = Variable(
                    torch.LongTensor(inst_data), volatile=self.test)
            if self.cuda:
                mask_tensor = mask_tensor.cuda()
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor, mask_tensor
        
        def pad_to_longest_cqs(insts, max_len):
            bs = len(insts)
            # batchsize * max_num * max_len
            inst_list = []
            mask_list = []

            for bi in range(bs):
                cq_insts = insts[bi] # max_num * len

                cq_inst = [inst[:max_len] + [0] * (max_len - len(inst[:max_len])) for inst in cq_insts] # max_num * max_len
                dim0 = len(cq_inst) # max_num
                dim1 = len(cq_inst[0]) # max_len

                cq_mask = [[0 for col in range(dim1)] for row in range(dim0)]

                for row in range(dim0):
                    for col in range(dim1):
                        if cq_inst[row][col] > 0:
                            cq_mask[row][col] = 1

                inst_list.append(cq_inst)
                mask_list.append(cq_mask)

            mask_tensor = Variable(
                torch.FloatTensor(mask_list), requires_grad=False)
            inst_tensor = Variable(
                torch.LongTensor(inst_list), volatile=self.test)
            if self.cuda:
                mask_tensor = mask_tensor.cuda()
                inst_tensor = inst_tensor.cuda()

            return inst_tensor, mask_tensor  # batchsize * max_num * max_len

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            inst_q, mask_q = pad_to_longest(self._inputs_q[start_idx:end_idx], MAX_Q_LEN)
            inst_d, mask_d = pad_to_longest(self._inputs_d[start_idx:end_idx], MAX_D_LEN)
            inst_dcq, mask_dcq = pad_to_longest_cqs(self._inputs_dcq[start_idx:end_idx], MAX_EXP_LEN)
            inst_dbody, mask_dbody = pad_to_longest(self._inputs_dbody[start_idx:end_idx], MAX_BODY_LEN)

            qid = self._inputs_qid[start_idx:end_idx]
            dsogouid = self._inputs_dsogouid[start_idx:end_idx]
            
            return inst_q, inst_d, inst_dcq, inst_dbody,\
                   mask_q, mask_d, mask_dcq, mask_dbody,\
                   qid, dsogouid

        else:

            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()
