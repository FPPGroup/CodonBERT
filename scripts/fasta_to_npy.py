import numpy as np
import argparse

def readFa(fa):
    with open(fa,'r') as FA:
        seqName,seq='',''
        while 1:
            line=FA.readline()
            line=line.strip('\n')
            if (line.startswith('>') or not line) and seqName:
                yield((seqName,seq))
            if line.startswith('>'):
                seqName = line[1:]
                seq=''
            else:
                seq+=line
            if not line:
                break
def set_args():
    parser = argparse.ArgumentParser(description='fasta to npy')
    parser.add_argument('-f', '--fasta_path', help='the seq fasta',required = True, type = str)
    parser.add_argument('-n', '--npy_path', help='the save path',required = True, type = str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    fasta_path = args.fasta_path
    npy_path = args.npy_path


    Seq = []
    for seqName,seq in readFa(fasta_path):
        Seq.append(seq)

    seq_dataset = np.array(Seq, dtype=object)
    np.save(npy_path, seq_dataset)
    # np.save(npy_path, Seq)