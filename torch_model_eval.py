import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUDA_VISIBLE_DEVICES']='1'
import time
import sys
import warnings
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

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

def GC_con(seq):
    length = len(seq)
    G_num = seq.count('G')
    C_num = seq.count('C')
    GC_content = (G_num+C_num)/length
    GC_content = round(GC_content, 4)
    return GC_content

############测试集240
# test_AA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/pep_kidney_high_TPM_encode_26_test.npy"
# test_DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_68_test.npy"
# test_mask_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_mask_test.npy"
test_AA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/pep_kidney_high_TPM_encode_25_240_test.npy"
test_DNA_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_67_240_test.npy"
test_mask_file_int = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/raw_data/TPM/train_data/csd_kidney_high_TPM_encode_mask_2567_240_test.npy"
test_AA_int = np.load(test_AA_file_int, allow_pickle=True)
test_DNA_int = np.load(test_DNA_file_int, allow_pickle=True)
test_mask = np.load(test_mask_file_int, allow_pickle=True)
#测试全部
test_AA_int = test_AA_int.astype(int)
test_DNA_int = test_DNA_int.astype(int)
test_mask_int = test_mask.astype(int)
#挑一条用来测试
# test_AA_int = test_AA_int[0].astype(int)
# test_DNA_int = test_DNA_int[0].astype(int)
# test_mask_int = test_mask[0].astype(int)
# test_AA_int = test_AA_int.reshape((1,-1))
# test_DNA_int = test_DNA_int.reshape((1,-1))
# test_mask_int = test_mask_int.reshape((1,-1))

# test_AA_int = test_AA_int[10].astype(int)
# test_DNA_int = test_DNA_int[10].astype(int)
# test_mask_int = test_mask[10].astype(int)
# test_AA_int = test_AA_int.reshape((1,-1))
# test_DNA_int = test_DNA_int.reshape((1,-1))
# test_mask_int = test_mask_int.reshape((1,-1))

test_AA_float_tensor = torch.tensor(test_AA_int, dtype=torch.int64)
# DNA_float_tensor = torch.tensor(DNA_int, dtype=torch.float32)
test_DNA_float_tensor = torch.tensor(test_DNA_int, dtype=torch.int64)
# DNA_float_tensor = torch.tensor(DNA_int, dtype=torch.float32)
test_mask_bool_tensor = torch.tensor(test_mask_int, dtype=torch.bool)
test_torch_dataset = data.TensorDataset(test_AA_float_tensor, test_DNA_float_tensor, test_mask_bool_tensor)
# test_train_iter = data.DataLoader(dataset=test_torch_dataset, batch_size=32, shuffle=True, num_workers=2)
# test_train_iter = data.DataLoader(dataset=test_torch_dataset, batch_size=1, shuffle=True, num_workers=2)
test_train_iter = data.DataLoader(dataset=test_torch_dataset, batch_size=1, shuffle=False, num_workers=2)

# log_dir = '/mnt/public2/jiangl/Projects/Project_plm_codon_optim/models/torch/20230419_epoch2-batch32-label67.pth'
# log_dir = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/models/torch/20230419-batch32-label67_150.pth"
# log_dir = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/models/torch/20230423-batch16-epoch200-label67-lrsch-randommask-2upmask_150.pth"
# log_dir = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/models/torch/20230423-batch10-label67-randommask_150.pth"
# log_dir = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/models/torch/20230428-batch32-label67-randommask_300.pth"
log_dir = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/models/torch/20230428-batch32-label67-randommask_320.pth"
# log_dir = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/models/torch/20230517-batch32-label67-randommask-GC_320.pth"
# log_dir = "/mnt/public2/jiangl/Projects/Project_plm_codon_optim/models/torch/20230517-batch32-label67-randommask-GC_200.pth"

model = torch.load(log_dir)
model = model.cpu()
# checkpoint = torch.load(log_dir)
# # model = model.load_state_dict(checkpoint['model_state_dict'])
# # optimizer = optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# model = checkpoint['model']
# optimizer = checkpoint['optimizer']
# epoch = checkpoint['epoch']
# model.eval()

AA_seq_result_all = []
DNA_annotation_result_all = []
AA_annotation_result_all = []
DNA_input_all = []
AA_input_all = []
DNA_acc_all = []
AA_acc_all = []
GC_con_all = []
for ind, (src, trg, mask) in enumerate(test_train_iter):
    # print(trg)
    zero_trg = trg.clone()
    zero_trg[:,:]=0
    # print(trg)
    # print(zero_trg)
    # src = src.to('cuda')
    # trg = trg.to('cuda')
    # mask = mask.to('cuda')
    # seq_logits, annotation_logits = model(src, trg, mask = mask)
    seq_logits, annotation_logits = model(src, zero_trg, mask = mask) #输入的DNA为全0tensor
    # seq_logits, annotation_logits = model.token_emb(src, zero_trg) #输出某一层的embedding 失败
    
    seq_logits = seq_logits[mask] #mask控制了哪些元素被保留，哪些元素被丢弃,mask为true的位置，相应的seq_logits元素保留，mask的shape为seq_logits[0],seq_logits[1],结果后的seq_logits形状为[(mask[0]*mask[1]),seq_logits[2]]
    seq_labels = src[mask]
    annotation_logits = annotation_logits[mask]
    annotation_labels = trg[mask]
    
    seq_logits_array = np.array(seq_logits.detach().numpy())#######执行之后
    result_seq = np.argmax(seq_logits_array, axis=1)#######执行之后
    annotation_logits_array = np.array(annotation_logits.detach().numpy())#######执行之后
    result_annotation = np.argmax(annotation_logits_array, axis=1)#######执行之后
    
    #seq的预测还原
    # AA_seq_result = seq_pre_to_AA(result_seq+3)
    AA_seq_result = seq_pre_to_AA(result_seq)
    #annotation的预测还原
    # DNA_annotation_result, AA_annotation_result = annotation_pre_to_AA(result_annotation+3)
    DNA_annotation_result, AA_annotation_result = annotation_pre_to_AA(result_annotation)
    
    #输入还原成真实序列
    # DNA_input, AA_input = annotation_pre_to_AA(annotation_labels)#数据集中的DNA序列不全是0 ##########################################
    AA_input = encode_AA_to_AA(seq_labels)#数据集中的输入的DNA序列全是0
   
    
    AA_encode_annotation_result = AA_to_encode_AA(AA_annotation_result)
    
    # #DNA序列acc计算
    # same_number = 0
    # single_acc = 0
    # for idx in range(len(annotation_labels)):
    #     if annotation_labels[idx] == result_annotation[idx]:
    #         same_number = same_number + 1
    # single_acc = same_number/len(annotation_labels)
    # DNA_acc_all.append(single_acc)
    
    #AA序列acc计算
    same_number = 0
    single_acc = 0
    for idx in range(len(AA_encode_annotation_result)):
        if seq_labels[idx] == AA_encode_annotation_result[idx]:
            same_number = same_number + 1
    single_acc = same_number/len(AA_encode_annotation_result)
    AA_acc_all.append(single_acc)
    
    #GC含量计算
    GC_con_single = GC_con(DNA_annotation_result)
    GC_con_all.append(GC_con_single)
    
    AA_seq_result_all.append(AA_seq_result)
    DNA_annotation_result_all.append(DNA_annotation_result)
    AA_annotation_result_all.append(AA_annotation_result)
    # DNA_input_all.append(DNA_input)##########################################
    AA_input_all.append(AA_input)
    
 
    
    
    # sys.exit(1)
#保存
#保存模型预测的DNA序列结果 保存成fasta格式
fasta_path ="/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/processed_output_data/fasta_file/epoch320_240_out.fasta" 
# fasta_path ="/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/processed_output_data/fasta_file/epoch320_5_out.fasta" 
fp = open(fasta_path,'w')
for i in range(len(DNA_annotation_result_all)):
    seq_name = ">seq_"+(str)(i)+"\n"
    seq = DNA_annotation_result_all[i]
    # print(seq_name)
    # print(seq)
    # sys.exit(1)
    fp.write(seq_name)
    fp.write(seq+"\n")


fp.close()

#输入氨基酸序列和seq预测的氨基酸序列结果比较
AA_input_compare_AA_seq_result = []
AA_input_compare_AA_seq_result_detail = []
AA_change = [[] for i in range (len(AA_input_all))]
length_all = 0
false_number = 0
for i in range(0, len(AA_input_all)):
    length_all = length_all + len(AA_input_all[i])
    if AA_input_all[i] == AA_seq_result_all[i]:
        AA_input_compare_AA_seq_result.append(True)
        AA_input_compare_AA_seq_result_detail.append(None)
    else:
        AA_input_compare_AA_seq_result.append(False)
        # print(AA_input_all[i])
        # print(AA_seq_result_all[i])
        single_detail = []
        for j in range(0, len(AA_input_all[i])):
            if AA_input_all[i][j]!=AA_seq_result_all[i][j]:
                # print('index:',j)
                # print('AA_seq:',AA_input_all[i][j])
                # print('AA_resul:',AA_seq_result_all[i][j])
                AA_change[i].append(AA_input_all[i][j])
                false_number = false_number + 1
                single_detail.append(str(j)+':'+str(AA_input_all[i][j])+'_'+str(AA_seq_result_all[i][j]))
        AA_input_compare_AA_seq_result_detail.append(single_detail)
    # sys.exit(1)
# print(AA_input_compare_AA_seq_result)
seq_AA_acc_metri = 1-(false_number/length_all)        

#输入氨基酸序列和annotation预测的氨基酸序列结果比较
AA_input_compare_AA_annotation_result = []
AA_input_compare_AA_annotation_result_detail = []
AA_change = [[] for i in range (len(AA_input_all))]
AA_change_idx = [[] for i in range (len(AA_input_all))]
AA_acc_metri = 0
length_all = 0
false_number = 0
for i in range(0, len(AA_input_all)):
    length_all = length_all + len(AA_input_all[i])
    if AA_input_all[i] == AA_annotation_result_all[i]:
        AA_input_compare_AA_annotation_result.append(True)
        AA_input_compare_AA_annotation_result_detail.append(None)
    else:
        AA_input_compare_AA_annotation_result.append(False)
        # print(AA_input_all[i])
        # print(AA_annotation_result_all[i])
        single_detail = []
        for j in range(0, len(AA_input_all[i])):
            if AA_input_all[i][j]!=AA_annotation_result_all[i][j]:
                # print('index:',j)
                # print('AA_annotation:',AA_input_all[i][j])
                # print('AA_resul:',AA_annotation_result_all[i][j])
                AA_change[i].append(AA_input_all[i][j])
                AA_change_idx[i].append(j)
                print(j)
                false_number = false_number + 1
                single_detail.append(str(j)+':'+str(AA_input_all[i][j])+'_'+str(AA_annotation_result_all[i][j]))
        AA_input_compare_AA_annotation_result_detail.append(single_detail)
AA_acc_metri = 1-(false_number/length_all)

#错误预测位点手动还原回去
#需要：输入的氨基酸序列、annotation还原的氨基酸序列，annotation还原的的DNA序列，预测错误的位点
AA_input_all #输入的氨基酸序列
AA_annotation_result_all #annotation还原的氨基酸序列
DNA_annotation_result_all #annotation还原的的DNA序列
AA_change_idx #预测错误的位点
fix_DNA_annotation_result_all = [[] for i in range (len(DNA_annotation_result_all))]

for seq_idx in range(len(AA_change_idx)):#有多少条序列
    result_DNA_seq = DNA_annotation_result_all[seq_idx]
    fix_DNA_one_codon = result_DNA_seq
    # print(len(result_DNA_seq))
    # print(result_DNA_seq)
    for change_idx in range(len(AA_change_idx[seq_idx])):
        false_local = AA_change_idx[seq_idx][change_idx]
        # print(false_local)#每条序列的错误预测位点
        # print('AA_input_all[seq_idx]:',AA_input_all[seq_idx])
        # print('AA_change_idx[seq_idx][change_idx]:',AA_change_idx[seq_idx][change_idx])
        input_AA_true = AA_input_all[seq_idx][AA_change_idx[seq_idx][change_idx]]
        # print(input_AA_true)#每条序列的错误预测位点的氨基酸，正确的氨基酸是哪个
        fix_codon = fix_AA_codon[input_AA_true]
        # print(fix_codon)#改成哪个密码子
        if false_local == 0:
            # pass
            fix_DNA_one_codon = fix_codon + fix_DNA_one_codon[((false_local+1)*3):]
        else:
            fix_DNA_one_codon = fix_DNA_one_codon[:((false_local*3))] + fix_codon + fix_DNA_one_codon[((false_local+1)*3):]
        # fix_DNA_annotation_result_all[seq_idx].append('fix后的整条氨基酸序列')
    fix_DNA_annotation_result_all[seq_idx].append(fix_DNA_one_codon)
    # print(len(fix_DNA_one_codon))
    # print('\n')
    

#保存
#保存模型预测结果修正过的DNA序列结果 保存成fasta格式
fasta_path ="/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/processed_output_data/fasta_file/epoch320_240_out_fix.fasta" 
# fasta_path ="/mnt/public2/jiangl/Projects/Project_plm_codon_optim/data/processed_output_data/fasta_file/epoch320_5_out_fix.fasta" 
fp = open(fasta_path,'w')
for i in range(len(fix_DNA_annotation_result_all)):
    seq_name = ">seq_"+(str)(i)+"\n"
    seq = fix_DNA_annotation_result_all[i]
    # print(seq_name)
    # print(seq[0])
    # sys.exit(1)
    fp.write(seq_name)
    fp.write(seq[0]+"\n")


fp.close()
