
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import argparse
import subprocess
import sys
from DataLoader import DataLoader
from DataLoaderTest import DataLoaderTest

from attnneudeftb import attnneudeftb
from cdssmtb import cdssmtb
from neudeftbtf import neudeftbtf
from neudeftb import neudeftb
from knrmtb import knrmtb
from cknrmtb import cknrmtb
from nrmftb import nrmftb

def test_epoch(model, test_data, file_name):
    out = open(file_name, "w")
    test_dict = dict()
    for batch in test_data:
        inst_q, inst_d, inst_dcq, inst_dbody, mask_q, mask_d, mask_dcq, mask_dbody, qid, dsogouid = batch

        outputs = model(inst_q, inst_d, inst_dcq, inst_dbody, mask_q, mask_d, mask_dcq, mask_dbody)
        output = outputs.data.tolist()
        tuples = zip(qid, dsogouid, output)
        for item in tuples:
            if item[0] not in test_dict:
                test_dict[item[0]] = []
            test_dict[item[0]].append((item[1], item[2]))
    for qid, value in test_dict.items():
        res = sorted(value, key=lambda x: x[1], reverse=True)
        for step, item in enumerate(res):
            out.write(str(qid) + " Q0 " + str(item[0]) + " " + str(step + 1) + " " + str(item[1]) + " knrm\n")
    out.close()

def eval_epoch(model, opt, crit, valid_data):
    model.eval()
    total_loss = 0
    for batch in valid_data:

        inst_q, inst_d_pos, inst_d_neg, inst_d_pos_cq, inst_d_neg_cq, inst_d_pos_body, inst_d_neg_body, \
        mask_q, mask_d_pos, mask_d_neg, mask_d_pos_cq, mask_d_neg_cq, mask_d_pos_body, mask_d_neg_body = batch

        outputs_pos = model(inst_q, inst_d_pos, inst_d_pos_cq, inst_d_pos_body, mask_q, mask_d_pos, mask_d_pos_cq,
                            mask_d_pos_body)
        outputs_neg = model(inst_q, inst_d_neg, inst_d_neg_cq, inst_d_neg_body, mask_q, mask_d_neg, mask_d_neg_cq,
                            mask_d_neg_body)

        label = torch.ones(outputs_pos.size())
        label = label.cuda() if opt.cuda else label

        batch_loss = crit(outputs_pos, outputs_neg, Variable(label, requires_grad=False))
        #total_loss += batch_loss.data[0]
        total_loss += batch_loss.data.item()

    return total_loss

def train(opt, crit, model, optimizer, train_data, valid_data, test_data):
    step = 0
    min_loss = float('inf')
    total_train_batch = len(train_data)
    total_valid_batch = len(valid_data)

    for epoch_i in range(opt.epoch):
        total_loss = 0.0

        for batch in train_data:
            step+=1

            optimizer.zero_grad()
            batch_loss = 0

            inst_q, inst_d_pos, inst_d_neg, inst_d_pos_cq, inst_d_neg_cq, inst_d_pos_body, inst_d_neg_body, \
            mask_q, mask_d_pos, mask_d_neg, mask_d_pos_cq, mask_d_neg_cq, mask_d_pos_body, mask_d_neg_body = batch

            outputs_pos = model(inst_q, inst_d_pos, inst_d_pos_cq, inst_d_pos_body, mask_q, mask_d_pos, mask_d_pos_cq, mask_d_pos_body)
            outputs_neg = model(inst_q, inst_d_neg, inst_d_neg_cq, inst_d_neg_body, mask_q, mask_d_neg, mask_d_neg_cq, mask_d_neg_body)

            optimizer.zero_grad()
            batch_loss = 0
            
            label = torch.ones(outputs_pos.size())
            label = label.cuda() if opt.cuda else label

            batch_loss += crit(outputs_pos, outputs_neg, Variable(label, requires_grad=False))
            
            # backward
            batch_loss.backward()
            
            # update parameters
            optimizer.step()

            #total_loss += batch_loss.data[0]
            total_loss += batch_loss.data.item()

            if step % opt.eval_step == 0:
                valid_loss = eval_epoch(model, opt, crit, valid_data)
                print(' Epoch %d Rate %f Training_loss %f Validation_loss %f' % (
                    epoch_i, step * 1.0 / total_train_batch, total_loss / opt.eval_step,
                    valid_loss / total_valid_batch))
                total_loss = 0

                file_name = opt.save_model + "_step_{}.trec".format(step)
                test_epoch(model, test_data, file_name)

                # test raw
                p = subprocess.Popen('python ./eval_top1.py -trec_file {0} -top1_file {1}'.format(file_name, opt.testraw),
                                     shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for line in p.stdout.readlines():
                    print line
                retval = p.wait()
                sys.stdout.flush()

                # test diff
                for k in [1, 10]:
                    p = subprocess.Popen(
                        'perl ./gdeval.pl -c -k {0} ./{1} {2}'.format(
                            k, opt.testdiff, file_name),
                        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    for line in p.stdout.readlines():
                        print opt.testdiff, k, line
                    retval = p.wait()
                    sys.stdout.flush()

                # save model
                if opt.save_model:
                    model_state_dict = model.state_dict()
                    checkpoint = {'model': model_state_dict}
                    if opt.save_mode == 'all':
                        model_name = opt.save_model + '_step_{}.chkpt'.format(step)
                        torch.save(checkpoint, model_name)
                    elif opt.save_mode == 'best':
                        model_name = opt.save_model + '.chkpt'
                        if valid_loss < min_loss:
                            min_loss = valid_loss
                            torch.save(checkpoint, model_name)
                            print('    - [Info] The checkpoint file has been updated.')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default="../data_cqbody.dat")
    parser.add_argument('-testdata', default="../head_testdata_cqbody.dat")
    parser.add_argument('-testraw', default="test_raw.qrel")
    parser.add_argument('-testdiff', default="test_diff.qrel")

    parser.add_argument('-model', default="7", help = "1:cdssmtb, 2:knrmtb, 3:cknrmtb, 4:nrmftb, 5:neudeftb, 6:neudeftbtf, 7:attnneudeftb")
    parser.add_argument('-load_emb', default="../word.emb.better") # init with knrm embedding, otherwise random init
    parser.add_argument('-cuda', type=bool, default=True)

    parser.add_argument('-eval_step', type=int, default=5000)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-vocab_size', default=200000)  # 165877

    parser.add_argument('-epoch', type=int, default=5)
    parser.add_argument('-save_model', default="model")
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-term_size', type=int, default=300)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-mu', type=list, default=[1, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9])
    parser.add_argument('-sigma', type=list, default=[1e-3] + [0.1] * 10)
    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('-n_head', type=int, default=4)

    opt = parser.parse_args()

    assert len(opt.mu) == len(opt.sigma)
    opt.n_bins = len(opt.mu)

    if opt.load_emb:
        opt.init_emb = torch.load(opt.load_emb)['word_emb']
    else:
        opt.init_emb = None

    if opt.model == '1':
        model = cdssmtb(opt)
        opt.save_model = "cdssmtb"
    elif opt.model == '2':
        model = knrmtb(opt)
        opt.save_model = "knrmtb"
    elif opt.model == '3':
        model = cknrmtb(opt)
        opt.save_model = "cknrmtb"
    elif opt.model == '4':
        model = nrmftb(opt)
        opt.save_model = "nrmftb"
    elif opt.model == '5':
        model = neudeftb(opt)
        opt.save_model = "neudeftb"
    elif opt.model == '6':
        model = neudeftbtf(opt)
        opt.save_model = "neudeftbtf"
    elif opt.model == '7':
        model = attnneudeftb(opt)
        opt.save_model = "attnneudeftb"


    print opt
    # the tensor in opt is init_emb

    crit = nn.MarginRankingLoss(margin=1, size_average=True)

    if opt.cuda:
        model = model.cuda()
        crit = crit.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, eps=1e-5)

    data = torch.load(opt.data)
    testdata = torch.load(opt.testdata)

    train_data = DataLoader(
        inputs_q=data['train']['q'],
        inputs_d_pos=data['train']['d_pos'],
        inputs_d_neg=data['train']['d_neg'],
        inputs_d_pos_cq=data['train']['d_pos_cq'],
        inputs_d_neg_cq=data['train']['d_neg_cq'],
        inputs_d_pos_body=data['train']['d_pos_body'],
        inputs_d_neg_body=data['train']['d_neg_body'],
        batch_size=opt.batch_size,
        cuda=opt.cuda)

    valid_data = DataLoader(
        inputs_q=data['valid']['q'],
        inputs_d_pos=data['valid']['d_pos'],
        inputs_d_neg=data['valid']['d_neg'],
        inputs_d_pos_cq=data['valid']['d_pos_cq'],
        inputs_d_neg_cq=data['valid']['d_neg_cq'],
        inputs_d_pos_body=data['valid']['d_pos_body'],
        inputs_d_neg_body=data['valid']['d_neg_body'],
        batch_size=opt.batch_size,
        shuffle=True,
        test=True,
        cuda=opt.cuda)

    test_data = DataLoaderTest(
        inputs_q=testdata['test']['q'],
        inputs_d=testdata['test']['d'],
        inputs_qid=testdata['test']['qid'],
        inputs_dsogouid=testdata['test']['dsogouid'],
        inputs_dcq=testdata['test']['dcq'],
        inputs_dbody=testdata['test']['dbody'],
        batch_size=opt.batch_size,
        test=True,
        cuda=opt.cuda)

    train(opt, crit, model, optimizer, train_data, valid_data, test_data)

if __name__ == "__main__":
    main()
