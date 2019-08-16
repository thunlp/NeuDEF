import argparse
import torch
import sys
from config import MAX_EXP_NUM, MAX_EXP_LEN

def cover_text2int(sentence):
    tokens = sentence.strip().split(",")
    return [int(token) for token in tokens]

def read_instances_from_file(inst_file):
    #out = str(q)+' '+str(dp)+' '+str(dn)+' '+str(dpcq)+' '+str(dncq)+' '+str(dpbody)+' '+str(dnbody)+'\n'

    q = []
    d_pos = []
    d_neg = []
    d_pos_cq = []
    d_neg_cq = []
    d_pos_body = []
    d_neg_body = []

    with open(inst_file) as f:
        for line in f:
            tokens = line.strip().split(' ')

            query = cover_text2int(tokens[0])
            dpos = cover_text2int(tokens[1])
            dneg = cover_text2int(tokens[2])

            dposexpall = []
            for tk in (tokens[3].split('#')):
                dposexpall.append(cover_text2int(tk))
            dnegexpall = []
            for tk in (tokens[4].split('#')):
                dnegexpall.append(cover_text2int(tk))

            while len(dposexpall) < MAX_EXP_NUM:
                dposexpall.append([0])
            dposexpall = dposexpall[0:MAX_EXP_NUM]

            while len(dnegexpall) < MAX_EXP_NUM:
                dnegexpall.append([0])
            dnegexpall = dnegexpall[0:MAX_EXP_NUM]

            dposbody = cover_text2int(tokens[5])
            dnegbody = cover_text2int(tokens[6])

            q.append(query)
            d_pos.append(dpos)
            d_neg.append(dneg)
            d_pos_cq.append(dposexpall)
            d_neg_cq.append(dnegexpall)
            d_pos_body.append(dposbody)
            d_neg_body.append(dnegbody)

    assert len(q)==len(d_pos)==len(d_neg)==len(d_pos_cq)==len(d_neg_cq)==len(d_pos_body)==len(d_neg_body)

    return q, d_pos, d_neg, d_pos_cq, d_neg_cq, d_pos_body, d_neg_body

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', default="sample_train_cqbody.cleaned")
    parser.add_argument('-valid', default="sample_valid_cqbody.cleaned")
    parser.add_argument('-save_data', default="sample_data_cqbody.dat")
    opt = parser.parse_args()

    valid_q, valid_d_pos, valid_d_neg, valid_d_pos_cq, valid_d_neg_cq, valid_d_pos_body, valid_d_neg_body = read_instances_from_file(opt.valid)

    train_q, train_d_pos, train_d_neg, train_d_pos_cq, train_d_neg_cq, train_d_pos_body, train_d_neg_body = read_instances_from_file(opt.train)

    data = {
        'settings': opt,
        'train': {
            'q': train_q,
            'd_pos': train_d_pos,
            'd_neg': train_d_neg,
            'd_pos_cq': train_d_pos_cq,
            'd_neg_cq': train_d_neg_cq,
            'd_pos_body': train_d_pos_body,
            'd_neg_body': train_d_neg_body,
        },
        'valid': {
            'q': valid_q,
            'd_pos': valid_d_pos,
            'd_neg': valid_d_neg,
            'd_pos_cq': valid_d_pos_cq,
            'd_neg_cq': valid_d_neg_cq,
            'd_pos_body': valid_d_pos_body,
            'd_neg_body': valid_d_neg_body,
        }
    }

    torch.save(data, opt.save_data)

if __name__ == '__main__':
    reload(sys)
    main()

