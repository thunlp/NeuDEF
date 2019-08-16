import argparse
import torch
import sys
from config import MAX_EXP_NUM

def cover_text2int(sentence):
    tokens = sentence.strip().split(",")
    return [int(token) for token in tokens]

def read_instances_from_file(inst_file):
    #out = str(q)+' '+str(d)+' '+str(qid)+' '+str(dsogouid)+' '+str(dcq)+' '+str(dbody)+'\n'

    q = []  # query
    d = []  # document title
    i = []  # query id
    n = []  # doc sogou number, ununique
    dcq = [] # documents' click query
    dbody = [] # document body

    with open(inst_file) as f:
        for line in f:
            tokens = line.strip().split(' ')

            q.append(cover_text2int(tokens[0]))
            d.append(cover_text2int(tokens[1]))
            i.append(tokens[2])
            n.append(tokens[3])

            dcqall = []
            for tk in (tokens[4].split('#')):
                dcqall.append(cover_text2int(tk))
            while len(dcqall) < MAX_EXP_NUM:
                dcqall.append([0])
            dcqall = dcqall[0:MAX_EXP_NUM]
            dcq.append(dcqall)
            
            dbody.append(cover_text2int(tokens[5]))

    assert len(q)==len(d)==len(i)==len(n)==len(dcq)==len(dbody)

    return q, d, i, n, dcq, dbody

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-inst_file', default="sample_test_with_ids_cqbody.cleaned")
    parser.add_argument('-save_data', default="sample_testdata_cqbody.dat")
    opt = parser.parse_args()

    q, d, qid, dsogouid, dcq, dbody = read_instances_from_file(opt.inst_file)

    data = {
        'settings': opt,
        'test': {
            'q': q,
            'd': d,
            'qid': qid,
            'dsogouid': dsogouid,
            'dcq': dcq,
            'dbody': dbody,
        }
    }

    torch.save(data, opt.save_data)

if __name__ == '__main__':
    reload(sys)
    main()
