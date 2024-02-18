import os
import warnings
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn.functional as FC
import torch

warnings.filterwarnings("ignore")

dict_raw_int = {'A': 3, 'B': 4, 'C': 5, 'D': 6, 'E': 7, 'F': 8, 'G': 9, 'H': 10, 'I': 11, 'J': 12, 'K': 13, 'L': 14, 'M': 15, 'N': 16, 'O': 17, 'P': 18, 'Q': 19, 'R': 20, 'S': 21, 'T': 22, 'U': 23, 'V': 24, 'W': 25, 'X': 26, 'Y': 27, 'Z': 28, 'a': 29, 'b': 30, 'c': 31, 'd': 32, 'e': 33, 'f': 34, 'g': 35, 'h': 36, 'i': 37, 'j': 38, 'k': 39, 'l': 40, 'm': 41, 'n': 42, 'o': 43, 'p': 44, 'q': 45, 'r': 46, 's': 47, 't': 48, 'u': 49, 'v': 50, 'w': 51, 'x': 52, 'y': 53, 'z': 54, '0': 55, '1': 56, '2': 57, '3': 58, '4': 59, '5': 60, '6': 61, '7': 62, '8': 63, '9': 64, ':': 65, ';': 66}
AA_dict_raw_int = {'A': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8, 'H': 9, 'I': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24}
codon_int = {'GGT':'a','GGC':'b','GGA':'c','GGG':'d','GCT':'e','GCC':'f','GCA':'g','GCG':'h','GTT':'i','GTC':'j','GTA':'k','GTG':'m','CTT':'l','CTC':'n','CTA':'o','CTG':'p','TTA':'q','TTG':'r','ATT':'s','ATC':'t','ATA':'u',
           'CCT':'v','CCA':'w','CCG':'x','CCC':'y','TTT':'z','TTC':'A','TAT':'B','TAC':'C','TGG':'D','TCT':'E','TCA':'F','TCC':'G','TCG':'H','AGT':'I','AGC':'J',
           'ACT':'K','ACC':'M','ACG':'L','ACA':'N','ATG':'O','TGT':'P','TGC':'Q','AAT':'R','AAC':'S','CAA':'T','CAG':'U','GAT':'V','GAC':'W',
           'GAA':'X','GAG':'Y','AAA':'Z','AAG':'1','CGT':'2','CGC':'3','CGG':'4','CGA':'5','AGA':'6','AGG':'7','CAT':'8','CAC':'9','TAA':'0','TAG':':','TGA':';'}
int_codon = dict((value, cod) for cod, value in codon_int.items())
G = ['GGT','GGC','GGA','GGG']
A = ['GCT','GCC','GCA','GCG']
V = ['GTT','GTC','GTA','GTG']
L = ['CTT','CTC','CTA','CTG','TTA','TTG']
I = ['ATT','ATC','ATA']
P = ['CCT','CCA','CCG','CCC']
F = ['TTT','TTC']
Y = ['TAT','TAC']
W = ['TGG']
S = ['TCT','TCA','TCC','TCG','AGT','AGC']
T = ['ACT','ACC','ACG','ACA']
M = ['ATG']
C = ['TGT','TGC']
N = ['AAT','AAC']
Q = ['CAA','CAG']
D = ['GAT','GAC']
E = ['GAA','GAG']
K = ['AAA','AAG']
R = ['CGT','CGC','CGG','CGA','AGA','AGG']
H = ['CAT','CAC']
X = ['TAA','TAG','TGA']
homonym_codon = {'G':G,'A':A,'V':V,'L':L,'I':I,'P':P,'F':F,'Y':Y,'W':W,'S':S,'T':T,'M':M,'C':C,'N':N,'Q':Q,'D':D,'E':E,'K':K,'R':R,'H':H,'X':X} 
fix_AA_codon = {'G':'GGC','A':'GCC','V':'GTC','L':'CTG','I':'ATC','P':'CCG','F':'TTC','Y':'TAC','W':'TGG','S':'AGC','T':'ACG','M':'ATG','C':'TGC','N':'AAC','Q':'CAG','D':'GAC','E':'GAG','K':'AAG','R':'CGC','H':'CAC','X':'TAG'} 


seq_len = 2048
is_binary = False
N_EPOCHS = 320
batch_size = 12
CodonBERT_path = '.'
log_dir = CodonBERT_path+"model/2024_0130/custom_kidney_CodonBert_model"
writer = SummaryWriter(CodonBERT_path+'logs/2023_0726/train_epoch320-batch12')
valid_writer = SummaryWriter(CodonBERT_path+'logs/2023_0726/valid_epoch320-batch12')
ACC_writer = SummaryWriter(CodonBERT_path+'logs/2023_0726/acc_epoch320-batch1')
tensorboard_ind = 0
valid_tensorboard_ind = 0
ACC_tensorboard_ind = 0


global device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def convert_list_to_dict(dict_raw_int,value):
    return [k for k, v in dict_raw_int.items() if v == (value)]

def DNA_to_AA(DNA_seq):
    AA_list = ""
    start = 0
    end = 3
    DNA_seq = DNA_seq.replace('U','T')
    while(end<=len(DNA_seq)+1):
        codon = DNA_seq[start:end]
        start+=3
        end+=3
        for AA,codons in homonym_codon.items():
            if codon in codons:
                AA_list += AA
    return AA_list

def seq_pre_to_AA(pre_list):
    AA_result = ''
    for i in range(len(pre_list)):
        AA_result=AA_result + str(convert_list_to_dict(AA_dict_raw_int,pre_list[i])).replace('[','').replace(']','').replace('\'','')
    return AA_result

def annotation_pre_to_AA(pre_list):
    raw_label = ''
    for i in range(len(pre_list)):
        raw_label=raw_label + str(convert_list_to_dict(dict_raw_int,pre_list[i])).replace('[','').replace(']','').replace('\'','')
    DNA_label = ''
    for i in range(len(raw_label)):
        DNA_label=DNA_label + str(convert_list_to_dict(codon_int,raw_label[i])).replace('[','').replace(']','').replace('\'','')
    AA_result = DNA_to_AA(DNA_label)
    return DNA_label, AA_result
    
def encode_AA_to_AA(pre_list):
    raw_label = ''
    for i in range(len(pre_list)):
        raw_label=raw_label + str(convert_list_to_dict(AA_dict_raw_int,pre_list[i])).replace('[','').replace(']','').replace('\'','')
    return raw_label
    
def AA_to_encode_AA(AA_list):
    encode_AA_label = []
    for i in range(len(AA_list)):
        encode_AA_label.append(AA_dict_raw_int[AA_list[i]])
    return encode_AA_label

def GC_con(seq):
    length = len(seq)
    G_num = seq.count('G')
    C_num = seq.count('C')
    GC_content = (G_num+C_num)/length
    GC_content = round(GC_content, 4)
    return GC_content

ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<PAD>', '<START>', '<END>']
n_aas = len(ALL_AAS)
aa_to_token_index = {aa: i+3 for i, aa in enumerate(ALL_AAS)}
additional_token_to_index = {token: i for i, token in enumerate(ADDITIONAL_TOKENS)}
unique_labels_list = [chr(i) for i in range(65, 91)] 
unique_labels_list = unique_labels_list + [chr(i) for i in range(97, 123)] 
unique_labels_list = unique_labels_list + [chr(i) for i in range(48, 60)]
label_to_index = {str(label): i+3 for i, label in enumerate(unique_labels_list)} 

def tokenize_seq(seq, max_len, aa_to_token_index, additional_token_to_index):
    coverted_seq  = [additional_token_to_index['<START>']] + [aa_to_token_index.get(aa, aa_to_token_index) for aa in seq] + [additional_token_to_index['<END>']]
    output_seq    = [additional_token_to_index['<PAD>'] for i in range(len(coverted_seq), max_len)]
    return coverted_seq + output_seq

def encode_seq_Y_68(seqs, seq_len, is_binary):
    unique_labels_list = [chr(i) for i in range(65, 91)] 
    unique_labels_list = unique_labels_list + [chr(i) for i in range(97, 123)] 
    unique_labels_list = unique_labels_list + [chr(i) for i in range(48, 60)]
    label_to_index = {str(label): i+3 for i, label in enumerate(unique_labels_list)} 

    Y = np.zeros((len(seqs), seq_len), dtype = int)
    sample_weigths = np.zeros((len(seqs), seq_len))
    
    for i, seq in enumerate(seqs):
        Y[i, 0] = 1
        Y[i, len(seq)+1] = 2
        for j, label in enumerate(seq):     
            Y[i, j + 1] = label_to_index[label]
        for r in range(len(seq)+2,seq_len):
            Y[i, r] = 0   
        sample_weigths[i, 1:(len(seq) + 1)] = 1    
    if is_binary:
        Y = np.expand_dims(Y, axis = -1)
        sample_weigths = np.expand_dims(sample_weigths, axis = -1)
    return Y, sample_weigths


def sample_DNA_to_int(DNA_seq,max_length):
    start = 0
    end = 3
    integer_encoded = []
    while(end<=len(DNA_seq)+1):
        codon = DNA_seq[start:end]
        start+=3
        end+=3
        integer_encoded.append(codon_int[codon])
    while(len(integer_encoded)<max_length):
        integer_encoded.append(0)
    return integer_encoded


def create_mask(seqs, seq_len):
    Y = np.zeros((len(seqs), seq_len), dtype = int)
    sample_weigths = np.zeros((len(seqs), seq_len))
    for i, seq in enumerate(seqs): 
        Y[i, 0] = 1
        Y[i, len(seq)+1] = 2
        for j, label in enumerate(seq):    
            Y[i, j + 1] = label_to_index[label]
        for r in range(len(seq)+2,seq_len):
            Y[i, r] = 0
        sample_weigths[i, 1:(len(seq) + 1)] = 1
    return sample_weigths


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


def self_evaluate(model, test_train_iter, valid_tensorboard_ind, valid_writer):
      for ind, (src, trg, mask) in enumerate(test_train_iter):
            src = src.to(device)
            trg = trg.to(device)
            mask = mask.to(device)
            zero_trg = trg.clone()
            zero_trg[:,:]=0
            seq_logits, annotation_logits = model(src, zero_trg, mask = mask) 
            seq_logits = seq_logits[mask] 
            seq_labels = src[mask]
            annotation_logits = annotation_logits[mask]
            annotation_labels = trg[mask]
            valid_seq_loss = FC.cross_entropy(seq_logits, seq_labels, reduction = 'mean')
            valid_annotation_loss = FC.cross_entropy(annotation_logits, annotation_labels, reduction = 'mean')
            valid_loss = valid_seq_loss + valid_annotation_loss

            valid_writer.add_scalar('valid_loss',valid_loss.item(),valid_tensorboard_ind)
            valid_writer.add_scalar('valid_seq_loss',valid_seq_loss.item(),valid_tensorboard_ind)
            valid_writer.add_scalar('valid_annotation_loss',valid_annotation_loss.item(),valid_tensorboard_ind)
            valid_tensorboard_ind += 1
      return valid_tensorboard_ind, valid_writer


def AA_acc(model, test_train_iter, ACC_tensorboard_ind, ACC_writer):
    AA_acc_all = 0
    for ind, (src, trg, mask) in enumerate(test_train_iter):
        src = src.to(device)
        trg = trg.to(device)
        mask = mask.to(device)
        zero_trg = trg.clone()
        zero_trg[:,:]=0
        seq_logits, annotation_logits = model(src, zero_trg, mask = mask) 
        seq_logits = seq_logits[mask]
        seq_labels = src[mask]
        annotation_logits = annotation_logits[mask]

        annotation_logits_array = np.array(annotation_logits.cpu().detach().numpy())
        result_annotation = np.argmax(annotation_logits_array, axis=1)
        DNA_annotation_result, AA_annotation_result = annotation_pre_to_AA(result_annotation)
        AA_encode_annotation_result = AA_to_encode_AA(AA_annotation_result)
        
        same_number = 0
        single_acc = 0
        for idx in range(len(AA_encode_annotation_result)):
            if seq_labels[idx] == AA_encode_annotation_result[idx]:
                same_number = same_number + 1
        single_acc = same_number/len(AA_encode_annotation_result)
        AA_acc_all = AA_acc_all + single_acc

        ACC_writer.add_scalar('AA_acc',single_acc,ACC_tensorboard_ind)
        ACC_tensorboard_ind += 1

    return ACC_tensorboard_ind, ACC_writer



