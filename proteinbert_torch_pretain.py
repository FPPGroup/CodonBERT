import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import time
import random
import warnings
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from Bio import SeqIO
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from protein_bert_pytorch import ProteinBERT, PretrainingWrapper
from tensorboardX import SummaryWriter
import torch.nn.functional as FC

### cuda2
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore")


# dict_raw_int = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,'i': 9,'j':10,'k':11,'l':12,'m':13,'n':14,'o':15,'p':16,'q':17,'r':18,'s':19,'t':20,'u':21,'v':22,'w':23,'x':24,'y':25,'z' :26, 'A' :27, 'B' :28, 'C' :29, 'D' :30, 'E' :31, 'F' :32, 'G' :33, 'H' :34, 'I' :35, 'J' :36, 'K' :37, 'L' :38, 'M' :39, 'N' :40, 'O' :41, 'P' :42, 'Q' :43, 'R' :44, 'S' :45 ,'T' :46 ,'U' :47 ,'V' :48 ,'W' :49 ,'X' :50 ,'Y' :51 ,'Z':52,'1':53,'2':54,'3':55,'4':56,'5':57,'6':58,'7':59,'8':60,'9':0,'0':61}
dict_raw_int = {'A': 3, 'B': 4, 'C': 5, 'D': 6, 'E': 7, 'F': 8, 'G': 9, 'H': 10, 'I': 11, 'J': 12, 'K': 13, 'L': 14, 'M': 15, 'N': 16, 'O': 17, 'P': 18, 'Q': 19, 'R': 20, 'S': 21, 'T': 22, 'U': 23, 'V': 24, 'W': 25, 'X': 26, 'Y': 27, 'Z': 28, 'a': 29, 'b': 30, 'c': 31, 'd': 32, 'e': 33, 'f': 34, 'g': 35, 'h': 36, 'i': 37, 'j': 38, 'k': 39, 'l': 40, 'm': 41, 'n': 42, 'o': 43, 'p': 44, 'q': 45, 'r': 46, 's': 47, 't': 48, 'u': 49, 'v': 50, 'w': 51, 'x': 52, 'y': 53, 'z': 54, '0': 55, '1': 56, '2': 57, '3': 58, '4': 59, '5': 60, '6': 61, '7': 62, '8': 63, '9': 64, ':': 65, ';': 66}
AA_dict_raw_int = {'A': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8, 'H': 9, 'I': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24}
codon_int = {'GGT':'a','GGC':'b','GGA':'c','GGG':'d','GCT':'e','GCC':'f','GCA':'g','GCG':'h','GTT':'i','GTC':'j','GTA':'k','GTG':'m','CTT':'l','CTC':'n','CTA':'o','CTG':'p','TTA':'q','TTG':'r','ATT':'s','ATC':'t','ATA':'u',
           'CCT':'v','CCA':'w','CCG':'x','CCC':'y','TTT':'z','TTC':'A','TAT':'B','TAC':'C','TGG':'D','TCT':'E','TCA':'F','TCC':'G','TCG':'H','AGT':'I','AGC':'J',
           'ACT':'K','ACC':'M','ACG':'L','ACA':'N','ATG':'O','TGT':'P','TGC':'Q','AAT':'R','AAC':'S','CAA':'T','CAG':'U','GAT':'V','GAC':'W',
           'GAA':'X','GAG':'Y','AAA':'Z','AAG':'1','CGT':'2','CGC':'3','CGG':'4','CGA':'5','AGA':'6','AGG':'7','CAT':'8','CAC':'9','TAA':'0','TAG':':','TGA':';'}
int_codon = dict((value, cod) for cod, value in codon_int.items())
# int_codon = dict[(value, cod) for cod, value in codon_int.items()]
#Create a homonym codon subtable
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


#还原蛋白质，和输入做比对
def convert_list_to_dict(dict_raw_int,value):
    # print('value:',value)
    # return [dict[str(i)] for i in lst]
    # return [k for k, v in dict_raw_int.items() if v == value]
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
        # print(codon)
        for AA,codons in homonym_codon.items():
            if codon in codons:
                AA_list += AA
    return AA_list

def seq_pre_to_AA(pre_list):
    #概率分布还原成字母
    AA_result = ''
    for i in range(len(pre_list)):
        AA_result=AA_result + str(convert_list_to_dict(AA_dict_raw_int,pre_list[i])).replace('[','').replace(']','').replace('\'','')
    return AA_result

def annotation_pre_to_AA(pre_list):
    #概率分布还原成字母
    raw_label = ''
    for i in range(len(pre_list)):
        raw_label=raw_label + str(convert_list_to_dict(dict_raw_int,pre_list[i])).replace('[','').replace(']','').replace('\'','')
    # print(raw_label)
    DNA_label = ''
    #字母还原成DNA
    for i in range(len(raw_label)):
        DNA_label=DNA_label + str(convert_list_to_dict(codon_int,raw_label[i])).replace('[','').replace(']','').replace('\'','')
    # print(DNA_label)
    #DNA还原成AA
    AA_result = DNA_to_AA(DNA_label)
    return DNA_label, AA_result
    # print(AA_result)
    
def encode_AA_to_AA(pre_list):
    #概率分布还原成氨基酸
    raw_label = ''
    for i in range(len(pre_list)):
        raw_label=raw_label + str(convert_list_to_dict(AA_dict_raw_int,pre_list[i])).replace('[','').replace(']','').replace('\'','')
    return raw_label
    
def AA_to_encode_AA(AA_list):
    encode_AA_label = []
    for i in range(len(AA_list)):
        # raw_label=raw_label + str(convert_list_to_dict(AA_dict_raw_int,pre_list[i])).replace('[','').replace(']','').replace('\'','')
        # print('AA_list[i]:',AA_list[i])
        encode_AA_label.append(AA_dict_raw_int[AA_list[i]])
    return encode_AA_label



model = ProteinBERT(
    # num_tokens = 21,
    # num_tokens = 24, ######################
    # num_tokens = 26,#21+4 start end pad other 0~25
    num_tokens = 25,#21+4 start end pad other 0~25
    # num_tokens = 22,#21+4 start end pad other 0~25


    # num_annotation = 8943,
    # num_annotation = 31000,
    # num_annotation = 500,
    # num_annotation = 120,
    # num_annotation = 122, ##############
    num_annotation = 512, ##############
    # num_annotation_class = 65, ##########
    # num_annotation_class = 62, ########## 4*4*4=62 包含终止密码子
    # num_annotation_class = 63, ########## 4*4*4=62 包含终止密码子 0~62
    # num_annotation_class = 68, ########## 4*4*4=64 64+4=68 (编码没有用other:64)包含终止密码子 0~62
    num_annotation_class = 67, ########## 4*4*4=64 64+4=68 (编码没有用other:64)包含终止密码子 0~62
    # num_annotation_class = 64, ########## 4*4*4=64 64+4=68 (编码没有用other:64)包含终止密码子 0~62


    dim = 512,
    # dim_global = 256,
    dim_global = 512,
    depth = 6,
    narrow_conv_kernel = 9,
    wide_conv_kernel = 9,
    wide_conv_dilation = 5,
    attn_heads = 8,
    attn_dim_head = 64,
    local_to_global_attn = False,
    local_self_attn = True,
    num_global_tokens = 2,
    glu_conv = False
)


# log_dir = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/models/torch/20230423-batch10-label67-randommask_150.pth"
# model = torch.load(log_dir)

# log_dir = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/models/torch/20230517-batch32-label67-randommask-GC_200.pth"
# model = torch.load(log_dir)

model = model.to('cuda')

# learner = PretrainingWrapper(
#     model,
#     random_replace_token_prob = 0.05,    # what percentage of the tokens to replace with a random one, defaults to 5% as in paper
#     remove_annotation_prob = 0.25,       # what percentage of annotations to remove, defaults to 25%
#     add_annotation_prob = 0.01,          # probability to add an annotation randomly, defaults to 1%
#     remove_all_annotations_prob = 0.5,   # what percentage of batch items to remove annotations for completely, defaults to 50%
#     seq_loss_weight = 1.,                # weight on loss of sequence
#     annotation_loss_weight = 1.,         # weight on loss of annotation
#     exclude_token_ids = (0, 1, 2)        # for excluding padding, start, and end tokens from being masked
# )
learner = PretrainingWrapper(
    model,
    # seq_length = 122, ######################
    seq_length = 512, ######################
    random_replace_token_prob = 0.05,    # what percentage of the tokens to replace with a random one, defaults to 5% as in paper
    remove_annotation_prob = 0.25,       # what percentage of annotations to remove, defaults to 25%
    add_annotation_prob = 0.01,          # probability to add an annotation randomly, defaults to 1%
    remove_all_annotations_prob = 0.5,   # what percentage of batch items to remove annotations for completely, defaults to 50%
    seq_loss_weight = 1.,                # weight on loss of sequence
    annotation_loss_weight = 1.,         # weight on loss of annotation
    # exclude_token_ids = (0, 1, 2)        # for excluding padding, start, and end tokens from being masked
    # exclude_token_ids = (25, 23, 24),
    # RNA_exclude_token_ids = (67, 65, 66)
    exclude_token_ids = (0, 1, 2),
    RNA_exclude_token_ids = (0, 1, 2)
    
)

# AA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/pep_kidney_high_TPM_encode_26_train.npy"
# DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_68_train.npy"
# # DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_skeletal_high_TPM_encode_62_train.npy"
# test_AA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/pep_kidney_high_TPM_encode_26_test.npy"
# test_DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_68_test.npy"#加了pad start end
# # test_DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_skeletal_high_TPM_encode_62_test.npy"
# train_mask_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_mask_train.npy"#加了pad start end
# test_mask_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_mask_test.npy"

# 81  164
# AA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/pep_kidney_high_TPM_encode_25_train.npy"
# DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_67_train.npy"
# # DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_skeletal_high_TPM_encode_62_train.npy"
# test_AA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/pep_kidney_high_TPM_encode_25_test.npy"
# test_DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_67_test.npy"#加了pad start end
# # test_DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_skeletal_high_TPM_encode_62_test.npy"
# train_mask_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_mask_2567_train.npy"#加了pad start end
# test_mask_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_mask_2567_test.npy"

#8000 240
AA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/pep_kidney_high_TPM_encode_25_8000_train.npy"
DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_67_8000_train.npy"
# DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_skeletal_high_TPM_encode_62_train.npy"
test_AA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/pep_kidney_high_TPM_encode_25_240_test.npy"
test_DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_67_240_test.npy"#加了pad start end
# test_DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_skeletal_high_TPM_encode_62_test.npy"
train_mask_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_mask_2567_8000_train.npy"#加了pad start end
test_mask_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_mask_2567_240_test.npy"

#对调seq annotation
# DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/pep_kidney_high_TPM_encode_26_train.npy"
# AA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_68_train.npy"
# # DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_skeletal_high_TPM_encode_62_train.npy"
# test_DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/pep_kidney_high_TPM_encode_26_test.npy"
# test_AA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_68_test.npy"#加了pad start end
# # test_DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_skeletal_high_TPM_encode_62_test.npy"
# train_mask_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_mask_train.npy"#加了pad start end
# test_mask_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_mask_test.npy"


AA_int = np.load(AA_file_int, allow_pickle=True)
DNA_int = np.load(DNA_file_int, allow_pickle=True)
test_AA_int = np.load(test_AA_file_int, allow_pickle=True)
test_DNA_int = np.load(test_DNA_file_int, allow_pickle=True)
train_mask = np.load(train_mask_file_int, allow_pickle=True)
test_mask = np.load(test_mask_file_int, allow_pickle=True)

#全部
# AA_int = AA_int.astype(int)
# # DNA_int = DNA_int.astype(float)
# DNA_int = DNA_int.astype(int)
# train_mask_int = train_mask.astype(int)
# test_AA_int = test_AA_int.astype(int)
# # DNA_int = DNA_int.astype(float)
# test_DNA_int = test_DNA_int.astype(int)
# test_mask_int = test_mask.astype(int)

#部分
AA_int = AA_int[0:2,:].astype(int)
# DNA_int = DNA_int.astype(float)
DNA_int = DNA_int[0:2,:].astype(int)
train_mask_int = train_mask[0:2,:].astype(int)
test_AA_int = test_AA_int[0:2,:].astype(int)
# DNA_int = DNA_int.astype(float)
test_DNA_int = test_DNA_int[0:2,:].astype(int)
test_mask_int = test_mask[0:2,:].astype(int)

AA_float_tensor = torch.tensor(AA_int, dtype=torch.int64)
# DNA_float_tensor = torch.tensor(DNA_int, dtype=torch.float32)
DNA_float_tensor = torch.tensor(DNA_int, dtype=torch.int64)
test_AA_float_tensor = torch.tensor(test_AA_int, dtype=torch.int64)
# DNA_float_tensor = torch.tensor(DNA_int, dtype=torch.float32)
test_DNA_float_tensor = torch.tensor(test_DNA_int, dtype=torch.int64)
train_mask_bool_tensor = torch.tensor(train_mask_int, dtype=torch.bool)
# DNA_float_tensor = torch.tensor(DNA_int, dtype=torch.float32)
test_mask_bool_tensor = torch.tensor(test_mask_int, dtype=torch.bool)

torch_dataset = data.TensorDataset(AA_float_tensor, DNA_float_tensor, train_mask_bool_tensor)
test_torch_dataset = data.TensorDataset(test_AA_float_tensor, test_DNA_float_tensor, test_mask_bool_tensor)
train_iter = data.DataLoader(dataset=torch_dataset, batch_size=32, shuffle=True, num_workers=2)
# test_train_iter = data.DataLoader(dataset=test_torch_dataset, batch_size=10, shuffle=True, num_workers=2)
test_train_iter = data.DataLoader(dataset=test_torch_dataset, batch_size=1, shuffle=True, num_workers=2)

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)###########################################
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

# writer = SummaryWriter('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/logs/proteinbert_pytorch/2023-04-15-epoch5-batch32-lr3e-4-label67-changeb2')
# writer = SummaryWriter('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/logs/proteinbert_pytorch/2023_0415/2023-04-15-epoch100-batch32-lr3e-4-label67-changeb2_addinput_50mask')
# writer = SummaryWriter('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/logs/proteinbert_pytorch/2023_0417/2023-04-17-epoch5-batch32-lr3e-4-label67-changeb2_addinput_randomtoken_10mask')
# writer = SummaryWriter('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/logs/proteinbert_pytorch/2023_0417/2023-04-17-epoch5-batch32-lr3e-4-label67-changeb2_noaddinput_randomtoken_10mask')
# writer = SummaryWriter('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/logs/proteinbert_pytorch/2023_0418/2023-04-18-epoch20-batch32-lr3e-4-label67-changeb2_noaddinput_randomtoken_10mask')
# writer = SummaryWriter('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/logs/proteinbert_pytorch/2023_0423/2023-04-23-epoch200-batch10-lr3e-4-label67_noaddinput_randomtoken_sum_upmask_2264_randommask')
# valid_writer = SummaryWriter('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/logs/proteinbert_pytorch/2023_0423/2023-04-23-epoch200-batch10-lr3e-4-label67_noaddinput_randomtoken_sum_upmask_2264_randommask_valid')
# writer = SummaryWriter('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/logs/proteinbert_pytorch/2023_0423/2023-04-23-epoch200-batch10-lr3e-4-label67_noaddinput_randomtoken_sum_upmask_randommask')
# valid_writer = SummaryWriter('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/logs/proteinbert_pytorch/2023_0423/2023-04-23-epoch200-batch10-lr3e-4-label67_noaddinput_randomtoken_sum_upmask_randommask_valid')
writer = SummaryWriter('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/logs/proteinbert_pytorch/2023_0428/2023-05-17-epoch320-batch32-lr3e-4-label67_noaddinput_randomtoken_sum_upmask_randommask_GC')
valid_writer = SummaryWriter('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/logs/proteinbert_pytorch/2023_0428/2023-05-17-epoch320-batch32-lr3e-4-label67_noaddinput_randomtoken_sum_upmask_randommask_valid_GC')
ACC_writer = SummaryWriter('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/logs/proteinbert_pytorch/2023_0428/2023-05-17-epoch320-batch1-lr3e-4-label67_noaddinput_randomtoken_sum_upmask_randommask_acc_GC')

# N_EPOCHS = 5
# N_EPOCHS = 10
# N_EPOCHS = 100
# N_EPOCHS = 20
# N_EPOCHS = 2
# N_EPOCHS = 30
# N_EPOCHS = 150
# N_EPOCHS = 200
# N_EPOCHS = 50
N_EPOCHS = 320
tensorboard_ind = 0
valid_tensorboard_ind = 0
ACC_tensorboard_ind = 0
# log_dir = './logs/ProteinBert_model.pth'
# log_dir = '/mnt/public2/jiangl/Projects/Project_plm_codon_optim/Project_RNA_deep_optim/models/protein_bert/protein_bert_torch/ProteinBert_model_2.pth'
# log_dir = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/models/epoch30-batch32-label63.pth"
log_dir = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/models/torch/20230517-batch32-label67-randommask-GC"

def self_evaluate(model, test_train_iter, valid_tensorboard_ind, valid_writer):
    #   model.eval()
    #   print('-----------------eval---------------------')
    #   for k,v in model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))
      for ind, (src, trg, mask) in enumerate(test_train_iter):
            src = src.to('cuda')
            trg = trg.to('cuda')
            mask = mask.to('cuda')
            zero_trg = trg.clone()
            zero_trg[:,:]=0
            seq_logits, annotation_logits = model(src, zero_trg, mask = mask) 
            seq_logits = seq_logits[mask] #mask控制了哪些元素被保留，哪些元素被丢弃,mask为true的位置，相应的seq_logits元素保留，mask的shape为seq_logits[0],seq_logits[1],结果后的seq_logits形状为[(mask[0]*mask[1]),seq_logits[2]]
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
    #   model.eval()
    #   print('-----------------eval---------------------')
    #   for k,v in model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))
    AA_acc_all = 0
    for ind, (src, trg, mask) in enumerate(test_train_iter):
        # print('ind:', ind)
        src = src.to('cuda')
        trg = trg.to('cuda')
        mask = mask.to('cuda')
        zero_trg = trg.clone()
        zero_trg[:,:]=0
        seq_logits, annotation_logits = model(src, zero_trg, mask = mask) 
        seq_logits = seq_logits[mask] #mask控制了哪些元素被保留，哪些元素被丢弃,mask为true的位置，相应的seq_logits元素保留，mask的shape为seq_logits[0],seq_logits[1],结果后的seq_logits形状为[(mask[0]*mask[1]),seq_logits[2]]
        seq_labels = src[mask]
        annotation_logits = annotation_logits[mask]

        annotation_logits_array = np.array(annotation_logits.cpu().detach().numpy())#######执行之后
        result_annotation = np.argmax(annotation_logits_array, axis=1)#######执行之后
        DNA_annotation_result, AA_annotation_result = annotation_pre_to_AA(result_annotation)
        AA_encode_annotation_result = AA_to_encode_AA(AA_annotation_result)
        
        #AA序列acc计算
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


for epoch in tqdm(range(N_EPOCHS)):
        # epoch = epoch + 200 #####################################################
        start_time = time.time()
        # total_loss = 0
        for ind, (src, trg, mask) in enumerate(train_iter):
        # for ind, (src, trg) in enumerate(train_iter):
            # print(src.shape)
            # print(trg.shape)
            
            optimizer.zero_grad()
            # output = model(src, trg)
            
            # output = output[1:].view(-1, output.shape[-1])              ## need fix
            # trg = trg[1:].view(-1)                                      ## need fix
            # loss = criterion(output, trg)
            
            # print(src.shape)
            # print(trg.shape)
            # print(mask.shape)
            # print('src',src)
            # print('trg',trg)
            # print('mask',mask)
            src = src.to('cuda')
            trg = trg.to('cuda')
            mask = mask.to('cuda')
            
            #生成随机数
            #tensor to numpy
            # mask_05 = mask_05.numpy()
            # # 插入false行
            # # mask_row = np.zeros((1,mask_05.shape[1]), dtype=bool)
            # mask_row = np.ones((1,mask_05.shape[1]), dtype=bool)
            # index_list = random.sample(range(0,mask_05.shape[0]-1),int(mask_05.shape[0]*0.5))
            # for index in index_list:
            #     mask_05[[index],:]=mask_row
            # mask_05 = torch.tensor(mask_05).to('cuda')
            # mask = mask
            
            
            # mask = torch.ones(src.shape[0], src.shape[1]).bool().to('cuda')
            
            
            # print(mask.shape)
            # loss =  (src, trg, mask = mask)
            # loss =  criterion(src, trg, mask)
            # loss =  criterion(src, trg, None)
            # loss, seq_loss, annotation_loss, seq_out, anno_out = learner(src, trg, mask = mask)
            # loss, seq_loss, annotation_loss = learner(src, trg, mask_05=mask_05, mask = mask)
            loss, seq_loss, annotation_loss, seq_logits, annotation_logits, seq_labels, annotation_labels = learner(src, trg, epoch, mask = mask)
            # total_loss = total_loss + loss
            writer.add_scalar('loss',loss.item(),tensorboard_ind)
            writer.add_scalar('seq_loss',seq_loss.item(),tensorboard_ind)
            writer.add_scalar('annotation_loss',annotation_loss.item(),tensorboard_ind)
            tensorboard_ind += 1
            
            loss.backward()
            # print('loss:',loss)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)    ## need fix
            optimizer.step()
        scheduler.step(loss)
        #验证
        valid_tensorboard_ind, valid_writer = self_evaluate(model, test_train_iter, valid_tensorboard_ind, valid_writer)
        
            # epoch_loss += loss.item()
        #AA_acc
        ACC_tensorboard_ind, ACC_writer = AA_acc(model, test_train_iter, ACC_tensorboard_ind, ACC_writer)
        

        # train_loss = train(model, train_iter, optimizer, criterion)
        # valid_loss = evaluate(model, train_iter, criterion)  # TODO: CHANGE valid_ITER
        # print('total_loss:',total_loss)                                                                                                                      
        end_time = time.time()
        if (epoch+1)==10 or (epoch+1)==20 or (epoch+1)==40 or (epoch+1)==60 or (epoch+1)==100 or (epoch+1)==120 or (epoch+1)==150 or (epoch+1)==200 or (epoch+1)==250 or (epoch+1)==300 or (epoch+1)==320:
             torch.save(model,log_dir+'_'+str(epoch+1)+'.pth')
        
        
        # epoch = epoch - 200 #####################################################
        # if (epoch+1+150)==200:
        #      torch.save(model,log_dir+'_'+str(epoch+1+150)+'.pth')
        # 保存模型
        # if (epoch+1)%10==0:
        #     state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        #     torch.save(state, log_dir)
print('-------------------finish--------------------')        


#         epoch_mins, epoch_secs = epoch_time(start_time, end_time)

#         print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
#         print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
#         print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')



# #data_load
# AA_file_int = "/mnt/public2/jiangl/Projects/Project_RNA_deep_optim/data/new_mRNA/mm_pep_less_8_int.npy"
# DNA_file_int = "/mnt/public2/jiangl/Projects/Project_RNA_deep_optim/data/new_mRNA/mm_cds_less_8_int.npy"
# AA_int = np.load(AA_file_int, allow_pickle=True)
# DNA_int = np.load(DNA_file_int, allow_pickle=True)

# AA_int = AA_int.astype(int)
# DNA_int = DNA_int.astype(float)

# AA_float_tensor = torch.tensor(AA_int, dtype=torch.int64)
# DNA_float_tensor = torch.tensor(DNA_int, dtype=torch.float32)

# torch_dataset = data.TensorDataset(AA_float_tensor, DNA_float_tensor)

# train_iter = data.DataLoader(dataset=torch_dataset, batch_size=64, shuffle=True, num_workers=2)

# criterion = nn.CrossEntropyLoss()
# # optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# # optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
# # optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
# # iterator = torch.utils.data.DataLoader

# N_EPOCHS = 50
# for epoch in tqdm(range(N_EPOCHS)):
#         start_time = time.time()
#         total_loss = 0
#         batch_num = 0
#         for _, (src, trg) in enumerate(train_iter):
#             # print(src.shape)
#             # print(trg.shape)
            
#             optimizer.zero_grad()
#             # output = model(src, trg)
            
#             # output = output[1:].view(-1, output.shape[-1])              ## need fix
#             # trg = trg[1:].view(-1)                                      ## need fix
#             # loss = criterion(output, trg)
            
            
#             src = src.to('cuda')
#             trg = trg.to('cuda')
            
#             mask = torch.ones(src.shape[0], src.shape[1]).bool().to('cuda')
#             # print(mask.shape)
#             # loss =  (src, trg, mask = mask)
#             # loss =  criterion(src, trg, mask)
#             # loss =  criterion(src, trg, None)
#             loss = learner(src, trg, mask = mask)
#             total_loss = total_loss + loss
#             batch_num = batch_num + 1
#             loss.backward()
#             print('batch_loss:',loss)
#             # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)    ## need fix
#             optimizer.step()
#             # epoch_loss += loss.item()

#         # train_loss = train(model, train_iter, optimizer, criterion)
#         # valid_loss = evaluate(model, train_iter, criterion)  # TODO: CHANGE valid_ITER
#         print('ave_loss:',total_loss/batch_num)
#         print('---------------------------------------------------------------------------')
#         end_time = time.time()

# #         epoch_mins, epoch_secs = epoch_time(start_time, end_time)

# #         print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
# #         print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
# #         print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')