import os
import numpy as np
import torch
import torch.utils.data as data
import argparse
from Codon_bert_pytorch import CodonBERT
from config_function import *

def set_args():
    parser = argparse.ArgumentParser(description='CodonBert:mRNA optimizaton')
    parser.add_argument('-m', '--model', help='the model path', required=True, type=str)
    parser.add_argument('-f', '--fasta', help='the seq fasta',required = True, type = str)             
    parser.add_argument('-o', '--output_path', help='the save path',required = True, type = str)
    args = parser.parse_args()      
    return args

args = set_args()
model_path = args.model
seq_file = args.fasta
model_output_save_path = args.output_path
model_output_fix_save_path = args.output_path.split('.')[0]+"_fix.fasta"

Seq_name = []
Seq = []
seq_num = 0
for seqName,seq in readFa(seq_file):
    Seq_name.append(seqName)
    Seq.append(seq)
    seqLen = len(seq)
    seq_num += 1
AA_seq_dataset = np.array(Seq, dtype=object)
AA_int_list = []
for idx in range(len(AA_seq_dataset)):
    AA_seq = tokenize_seq(AA_seq_dataset[idx], seq_len, aa_to_token_index, additional_token_to_index)
    AA_int_list.append(AA_seq)

test_AA_int = np.array(AA_int_list)
test_mask = create_mask(AA_seq_dataset)
test_AA_int = test_AA_int.astype(int)
test_DNA_int = torch.zeros((test_AA_int.shape[0], test_AA_int.shape[1]))
test_mask_int = test_mask.astype(int)
test_AA_float_tensor = torch.tensor(test_AA_int, dtype=torch.int64)
test_DNA_float_tensor = torch.tensor(test_DNA_int, dtype=torch.int64)
test_mask_bool_tensor = torch.tensor(test_mask_int, dtype=torch.bool)
test_torch_dataset = data.TensorDataset(test_AA_float_tensor, test_DNA_float_tensor, test_mask_bool_tensor)
test_train_iter = data.DataLoader(dataset=test_torch_dataset, batch_size=1, shuffle=False, num_workers=2)

model = CodonBERT(
    num_tokens = 25,
    num_annotation_class = 67,
    dim = 128,
    dim_global = 128,
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
model.load_state_dict(torch.load(model_path))

model = model.cpu()

AA_seq_result_all = []
DNA_annotation_result_all = []
AA_annotation_result_all = []
DNA_input_all = []
AA_input_all = []
DNA_acc_all = []
AA_acc_all = []
GC_con_all = []
for ind, (src, trg, mask) in enumerate(test_train_iter):
    zero_trg = trg.clone()
    zero_trg[:,:]=0
    seq_logits, annotation_logits = model(src, zero_trg, mask = mask) 
    seq_logits = seq_logits[mask] 
    seq_labels = src[mask]
    annotation_logits = annotation_logits[mask]
    annotation_labels = trg[mask]
    seq_logits_array = np.array(seq_logits.detach().numpy())
    result_seq = np.argmax(seq_logits_array, axis=1)
    annotation_logits_array = np.array(annotation_logits.detach().numpy())
    result_annotation = np.argmax(annotation_logits_array, axis=1)
    AA_seq_result = seq_pre_to_AA(result_seq)
    DNA_annotation_result, AA_annotation_result = annotation_pre_to_AA(result_annotation)
    AA_input = encode_AA_to_AA(seq_labels)
    AA_encode_annotation_result = AA_to_encode_AA(AA_annotation_result)
    same_number = 0
    single_acc = 0
    for idx in range(len(AA_encode_annotation_result)):
        if seq_labels[idx] == AA_encode_annotation_result[idx]:
            same_number = same_number + 1
    single_acc = same_number/len(AA_encode_annotation_result)
    AA_acc_all.append(single_acc)
    GC_con_single = GC_con(DNA_annotation_result)
    GC_con_all.append(GC_con_single)
    AA_seq_result_all.append(AA_seq_result)
    DNA_annotation_result_all.append(DNA_annotation_result)
    AA_annotation_result_all.append(AA_annotation_result)
    AA_input_all.append(AA_input)
    
fasta_path = model_output_save_path
fp = open(fasta_path,'w')
for i in range(len(DNA_annotation_result_all)):
    seq_name = '>'+Seq_name[i]+"\n"
    seq = DNA_annotation_result_all[i]
    fp.write(seq_name)
    fp.write(seq+"\n")
fp.close()    

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
        single_detail = []
        for j in range(0, len(AA_input_all[i])):
            if AA_input_all[i][j]!=AA_annotation_result_all[i][j]:
                AA_change[i].append(AA_input_all[i][j])
                AA_change_idx[i].append(j)
                false_number = false_number + 1
                single_detail.append(str(j)+':'+str(AA_input_all[i][j])+'_'+str(AA_annotation_result_all[i][j]))
        AA_input_compare_AA_annotation_result_detail.append(single_detail)
AA_acc_metri = 1-(false_number/length_all)

fix_DNA_annotation_result_all = [[] for i in range (len(DNA_annotation_result_all))]

for seq_idx in range(len(AA_change_idx)):
    result_DNA_seq = DNA_annotation_result_all[seq_idx]
    fix_DNA_one_codon = result_DNA_seq
    for change_idx in range(len(AA_change_idx[seq_idx])):
        false_local = AA_change_idx[seq_idx][change_idx]
        input_AA_true = AA_input_all[seq_idx][AA_change_idx[seq_idx][change_idx]]
        fix_codon = fix_AA_codon[input_AA_true]
        if false_local == 0:
            fix_DNA_one_codon = fix_codon + fix_DNA_one_codon[((false_local+1)*3):]
        else:
            fix_DNA_one_codon = fix_DNA_one_codon[:((false_local*3))] + fix_codon + fix_DNA_one_codon[((false_local+1)*3):]
    fix_DNA_annotation_result_all[seq_idx].append(fix_DNA_one_codon)


fasta_path = model_output_fix_save_path
fp = open(fasta_path,'w')
for i in range(len(fix_DNA_annotation_result_all)):
    seq_name = '>'+Seq_name[i]+"\n"
    seq = fix_DNA_annotation_result_all[i]
    fp.write(seq_name)
    fp.write(seq[0]+"\n")
fp.close()

