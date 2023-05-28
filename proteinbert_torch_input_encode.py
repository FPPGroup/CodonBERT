#cds_seq_to_torch_input
import numpy as np
import pandas as pd
import os


ALL_AAS = 'ACDEFGHIKLMNPQRSTUVWXY'
ADDITIONAL_TOKENS = ['<PAD>', '<START>', '<END>']

# Each sequence is added <START> and <END> tokens

n_aas = len(ALL_AAS)
aa_to_token_index = {aa: i+3 for i, aa in enumerate(ALL_AAS)}
additional_token_to_index = {token: i for i, token in enumerate(ADDITIONAL_TOKENS)}

unique_labels_list = [chr(i) for i in range(65, 91)] # 所有大写字母
unique_labels_list = unique_labels_list + [chr(i) for i in range(97, 123)] # 所有小写字母
unique_labels_list = unique_labels_list + [chr(i) for i in range(48, 60)]
# label_to_index = {str(label): i for i, label in enumerate(unique_labels_list)} # 生成字典
label_to_index = {str(label): i+3 for i, label in enumerate(unique_labels_list)} # 生成字典

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
codon_int = {'GGT':'a','GGC':'b','GGA':'c','GGG':'d','GCT':'e','GCC':'f','GCA':'g','GCG':'h','GTT':'i','GTC':'j','GTA':'k','GTG':'m','CTT':'l','CTC':'n','CTA':'o','CTG':'p','TTA':'q','TTG':'r','ATT':'s','ATC':'t','ATA':'u',
           'CCT':'v','CCA':'w','CCG':'x','CCC':'y','TTT':'z','TTC':'A','TAT':'B','TAC':'C','TGG':'D','TCT':'E','TCA':'F','TCC':'G','TCG':'H','AGT':'I','AGC':'J',
           'ACT':'K','ACC':'M','ACG':'L','ACA':'N','ATG':'O','TGT':'P','TGC':'Q','AAT':'R','AAC':'S','CAA':'T','CAG':'U','GAT':'V','GAC':'W',
           'GAA':'X','GAG':'Y','AAA':'Z','AAG':'1','CGT':'2','CGC':'3','CGG':'4','CGA':'5','AGA':'6','AGG':'7','CAT':'8','CAC':'9','TAA':'0','TAG':':','TGA':';'}
int_codon = dict((value, cod) for cod, value in codon_int.items())

def tokenize_seq(seq, max_len, aa_to_token_index, additional_token_to_index):
    coverted_seq  = [additional_token_to_index['<START>']] + [aa_to_token_index.get(aa, aa_to_token_index) for aa in seq] + [additional_token_to_index['<END>']]
    output_seq    = [additional_token_to_index['<PAD>'] for i in range(len(coverted_seq), max_len)]
    return coverted_seq + output_seq

def encode_seq_Y_68(seqs, seq_len, is_binary):
    unique_labels_list = [chr(i) for i in range(65, 91)] # 所有大写字母
    unique_labels_list = unique_labels_list + [chr(i) for i in range(97, 123)] # 所有小写字母
    # unique_labels_list = unique_labels_list + [chr(i) for i in range(48, 58)]
    unique_labels_list = unique_labels_list + [chr(i) for i in range(48, 60)]
    # label_to_index = {str(label): i for i, label in enumerate(unique_labels_list)} # 生成字典
    label_to_index = {str(label): i+3 for i, label in enumerate(unique_labels_list)} # 生成字典
    print(label_to_index)
    # sys.exit(1)

    Y = np.zeros((len(seqs), seq_len), dtype = int)
    sample_weigths = np.zeros((len(seqs), seq_len))
    
    for i, seq in enumerate(seqs): # seqs = [[seq1],[seq2],[seq]] 
        Y[i, 0] = 1
        Y[i, len(seq)+1] = 2
        
        for j, label in enumerate(seq):     # seq = "wqeqweqw"
            Y[i, j + 1] = label_to_index[label]
            

        for r in range(len(seq)+2,seq_len):
            Y[i, r] = 0
            
        sample_weigths[i, 1:(len(seq) + 1)] = 1
        
    if is_binary:
        Y = np.expand_dims(Y, axis = -1)
        sample_weigths = np.expand_dims(sample_weigths, axis = -1)
    
    return Y, sample_weigths

#一个密码子转变成指定长度的数字编码，非onehot
def sample_DNA_to_int(DNA_seq,max_length):
    start = 0
    end = 3
    integer_encoded = []
    count_list = 0
    # integer_encoded.append(1)
    while(end<=len(DNA_seq)+1):
        codon = DNA_seq[start:end]
        start+=3
        end+=3
        # integer_encoded.append(one_to_63(codon_int[codon],62))
        integer_encoded.append(codon_int[codon])
    # integer_encoded.append(2)
    while(len(integer_encoded)<max_length):
        integer_encoded.append(0)
    return integer_encoded

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

seq_len = 512
is_binary = False
DNA_file = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM.npy"
AA_file = DNA_file.replace('cds','pep')
DNA_seq_dataset = np.load(DNA_file,allow_pickle=True)

# DNA to AA
AA_list = []
for idx in range(len(DNA_seq_dataset)):
    DNA_seq_dataset[idx] = DNA_seq_dataset[idx].replace('U','T')
    AA_seq = DNA_to_AA(DNA_seq_dataset[idx])
    AA_list.append(AA_seq) 
AA_seq_dataset = np.array(AA_list,dtype=object)

# RNA to zm
# DNA_seq_dataset = np.load(DNA_file,allow_pickle=True)
int_list = []
for idx in range(len(DNA_seq_dataset)):
    DNA_seq_dataset[idx] = DNA_seq_dataset[idx].replace('U','T')
    # int_seq = DNA_to_int(DNA_seq_dataset[idx],31000)
    # int_seq = DNA_to_int(DNA_seq_dataset[idx],0)
    # int_seq = sample_DNA_to_int(DNA_seq_dataset[idx],120)
    int_seq = sample_DNA_to_int(DNA_seq_dataset[idx],0)
    int_seq = str(int_seq).replace('[','').replace(']','').replace(',','').replace(' ','').replace('\'','')
    # int_seq = DNA_to_int(DNA_seq_dataset[idx],0)
    int_list.append(int_seq)
    
DNA_int_result, sample_weigths = encode_seq_Y_68(int_list,seq_len,is_binary)

#蛋白质数据集编码
# DNA_seq_dataset = np.load(DNA_file,allow_pickle=True)
AA_int_list = []
for idx in range(len(AA_seq_dataset)):
    # AA_seq_dataset[idx] = DNA_seq_dataset[idx].replace('U','T')
    # AA_seq = DNA_to_AA(AA_seq_dataset[idx])
    AA_seq = tokenize_seq(AA_seq_dataset[idx], max_len, aa_to_token_index, additional_token_to_index)
    AA_int_list.append(AA_seq)


#############划分训练集，验证集和测试集
# #划分氨基酸序列
dataset_length = len(AA_int_list)
train_end = int(dataset_length*0.98)
train_AA = AA_int_list[0:train_end]
test_AA = AA_int_list[train_end:dataset_length]


#划分DNA序列
dataset_length = len(DNA_int_result)
train_end = int(dataset_length*0.98)
train_DNA = DNA_int_result[0:train_end]
test_DNA = DNA_int_result[train_end:dataset_length]

#划分mask
dataset_length = len(sample_weigths)
train_end = int(dataset_length*0.98)
train_mask = sample_weigths[0:train_end]
test_mask = sample_weigths[train_end:dataset_length]

np.save('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_67_train.npy',train_DNA)
np.save('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/pep_kidney_high_TPM_encode_25_train.npy',train_AA)
np.save('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_67_test.npy',test_DNA)
np.save('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/pep_kidney_high_TPM_encode_25_test.npy',test_AA)
np.save('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_mask_2567_train.npy',train_mask)
np.save('/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_mask_2567_test.npy',test_mask)

