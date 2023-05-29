import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from protein_bert_pytorch import ProteinBERT, PretrainingWrapper
from config_function import *
import argparse

warnings.filterwarnings("ignore")

model = ProteinBERT(
    num_tokens = 25,
    num_annotation = 512, 
    num_annotation_class = 67, 
    dim = 512,
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
model = model.to('cuda')

learner = PretrainingWrapper(
    model,
    seq_length = 512, 
    random_replace_token_prob = 0.05,    
    remove_annotation_prob = 0.25,       
    add_annotation_prob = 0.01,          
    remove_all_annotations_prob = 0.5,   
    seq_loss_weight = 1.,                
    annotation_loss_weight = 1.,         
    exclude_token_ids = (0, 1, 2),
    RNA_exclude_token_ids = (0, 1, 2) 
)

def set_args():
    parser = argparse.ArgumentParser(description='CodonBert:model train')      
    parser.add_argument('-i', '--input', help='the mRNA seq file <.npy>',required = True, type = str)             
    args = parser.parse_args()      
    return args

args = set_args()
DNA_file = args.input

AA_file = DNA_file.replace('cds','pep')
DNA_seq_dataset = np.load(DNA_file,allow_pickle=True)
AA_list = []
for idx in range(len(DNA_seq_dataset)):
    DNA_seq_dataset[idx] = DNA_seq_dataset[idx].replace('U','T')
    AA_seq = DNA_to_AA(DNA_seq_dataset[idx])
    AA_list.append(AA_seq) 
AA_seq_dataset = np.array(AA_list,dtype=object)
int_list = []
for idx in range(len(DNA_seq_dataset)):
    DNA_seq_dataset[idx] = DNA_seq_dataset[idx].replace('U','T')
    int_seq = sample_DNA_to_int(DNA_seq_dataset[idx],0)
    int_seq = str(int_seq).replace('[','').replace(']','').replace(',','').replace(' ','').replace('\'','')
    int_list.append(int_seq)  
DNA_int_result, sample_weigths = encode_seq_Y_68(int_list,seq_len,is_binary)
AA_int_list = []
for idx in range(len(AA_seq_dataset)):
    AA_seq = tokenize_seq(AA_seq_dataset[idx], seq_len, aa_to_token_index, additional_token_to_index)
    AA_int_list.append(AA_seq)
dataset_length = len(AA_int_list)
train_end = int(dataset_length*0.98)
train_AA = AA_int_list[0:train_end]
test_AA = AA_int_list[train_end:dataset_length]
dataset_length = len(DNA_int_result)
train_end = int(dataset_length*0.98)
train_DNA = DNA_int_result[0:train_end]
test_DNA = DNA_int_result[train_end:dataset_length]
dataset_length = len(sample_weigths)
train_end = int(dataset_length*0.98)
train_mask = sample_weigths[0:train_end]
test_mask = sample_weigths[train_end:dataset_length]
AA_int = np.array(train_AA)
DNA_int = np.array(train_DNA)
test_AA_int = np.array(test_AA)
test_DNA_int = np.array(test_DNA)
train_mask = train_mask
test_mask = test_mask

AA_int = AA_int.astype(int)
DNA_int = DNA_int.astype(int)
train_mask_int = train_mask.astype(int)
test_AA_int = test_AA_int.astype(int)
test_DNA_int = test_DNA_int.astype(int)
test_mask_int = test_mask.astype(int)

AA_float_tensor = torch.tensor(AA_int, dtype=torch.int64)
DNA_float_tensor = torch.tensor(DNA_int, dtype=torch.int64)
test_AA_float_tensor = torch.tensor(test_AA_int, dtype=torch.int64)
test_DNA_float_tensor = torch.tensor(test_DNA_int, dtype=torch.int64)
train_mask_bool_tensor = torch.tensor(train_mask_int, dtype=torch.bool)
test_mask_bool_tensor = torch.tensor(test_mask_int, dtype=torch.bool)

torch_dataset = data.TensorDataset(AA_float_tensor, DNA_float_tensor, train_mask_bool_tensor)
test_torch_dataset = data.TensorDataset(test_AA_float_tensor, test_DNA_float_tensor, test_mask_bool_tensor)
train_iter = data.DataLoader(dataset=torch_dataset, batch_size=32, shuffle=True, num_workers=2)
test_train_iter = data.DataLoader(dataset=test_torch_dataset, batch_size=1, shuffle=True, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)


for epoch in tqdm(range(N_EPOCHS)):
        start_time = time.time()
        for ind, (src, trg, mask) in enumerate(train_iter):      
            optimizer.zero_grad()
            src = src.to('cuda')
            trg = trg.to('cuda')
            mask = mask.to('cuda') 
            loss = learner(src, trg, epoch, mask = mask)
            loss.backward()
            optimizer.step()
        scheduler.step(loss)                                                                                                                    
        end_time = time.time()
        if (epoch+1)==N_EPOCHS:
             torch.save(model,log_dir+'_'+str(N_EPOCHS)+'.pth')
        
