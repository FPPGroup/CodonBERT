import gzip
from Bio.Seq import Seq
from Bio import SeqIO
import pandas as pd
import re
import seaborn as sns
from collections import defaultdict
import random
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys
import os
from get_metrics import *
import argparse

# Create a homonym codon subtable
G = ['GGT', 'GGC', 'GGA', 'GGG']
A = ['GCT', 'GCC', 'GCA', 'GCG']
V = ['GTT', 'GTC', 'GTA', 'GTG']
L = ['CTT', 'CTC', 'CTA', 'CTG', 'TTA', 'TTG']
I = ['ATT', 'ATC', 'ATA']
P = ['CCT', 'CCA', 'CCG', 'CCC']
F = ['TTT', 'TTC']
Y = ['TAT', 'TAC']
W = ['TGG']
S = ['TCT', 'TCA', 'TCC', 'TCG', 'AGT', 'AGC']
T = ['ACT', 'ACC', 'ACG', 'ACA']
M = ['ATG']
C = ['TGT', 'TGC']
N = ['AAT', 'AAC']
Q = ['CAA', 'CAG']
D = ['GAT', 'GAC']
E = ['GAA', 'GAG']
K = ['AAA', 'AAG']
R = ['CGT', 'CGC', 'CGG', 'CGA', 'AGA', 'AGG']
H = ['CAT', 'CAC']
X = ['TAA', 'TAG', 'TGA']
homonym_codon = {'G': G, 'A': A, 'V': V, 'L': L, 'I': I, 'P': P, 'F': F, 'Y': Y, 'W': W, 'S': S, 'T': T, 'M': M, 'C': C,
                 'N': N, 'Q': Q, 'D': D, 'E': E, 'K': K, 'R': R, 'H': H, 'X': X}


def DNA_to_AA(DNA_seq):
    AA_list = ""
    start = 0
    end = 3
    DNA_seq = DNA_seq.replace('U', 'T')
    while (end <= len(DNA_seq) + 1):
        codon = DNA_seq[start:end]
        start += 3
        end += 3
        for AA, codons in homonym_codon.items():
            if codon in codons:
                AA_list += AA
    return AA_list


def readFa(fa):
    '''
    @msg: Read a fasta file
    @param fa {str}  Fasta file path
    @return: {generator} Returns a generator that can iterate over each sequence name and sequence in the fasta file
    '''
    with open(fa, 'r') as FA:
        seqName, seq = '', ''
        while 1:
            line = FA.readline()
            line = line.strip('\n')
            if (line.startswith('>') or not line) and seqName:
                yield ((seqName, seq))
            if line.startswith('>'):
                seqName = line[1:]
                seq = ''
            else:
                seq += line
            if not line:
                break


def dna_to_amino_acid(dna_sequence):
    coding_dna = Seq(dna_sequence)
    return str(coding_dna.translate())


def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def set_args():
    parser = argparse.ArgumentParser(description='CodonBert:model train')
    parser.add_argument('-e', '--env_path', help='environment absolute path', required=True, type=str)
    parser.add_argument('-t', '--tsv_path', help='transcript_rna_tissue.tsv file <.tsv>', required=True, type=str)
    parser.add_argument('-l', '--lations', help='gencode.v43.pc_translations.fa.gz file <.gz>', required=True, type=str)
    parser.add_argument('-c', '--scripts', help='gencode.v43.pc_transcripts.fa.gz file <.gz>', required=True, type=str)
    parser.add_argument('-o', '--output_path', help='result save path', required=True, type=str)
    args = parser.parse_args()
    return args


args = set_args()
env_path = args.env_path
tsv_path = args.tsv_path
AA_fa_gz_path = args.lations
DNA_fa_gz_path = args.scripts
result_save_path = args.output_path
tissue_TPM = pd.read_csv(tsv_path, sep='\t')
directory_path = './data/train_data'
if (os.path.exists(directory_path) == False):
    os.makedirs(directory_path)

# Retrieve the enstid with high TPM values for a specific tissue
tpm_val = tissue_TPM.loc[:, tissue_TPM.columns.str.contains('kidney')]
tpm_val['TPM_mean'] = tpm_val.apply(lambda x: np.median(x), axis=1)
merged = pd.concat([tissue_TPM.loc[:, ['ensgid', 'enstid']], tpm_val], axis=1)
kidney_beyong5_id = merged.loc[merged.TPM_mean > 5, ['ensgid', 'enstid']]
kidney_beyong5_id = kidney_beyong5_id.reset_index(drop=True)
kidney_beyong5_id_part = kidney_beyong5_id.loc[0:10, :]

# Read compressed FASTA file
AA_fa_dict = {}
DNA_fa_dict = {}

with gzip.open(AA_fa_gz_path, "rt") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        ENSG_result = re.findall(r'\bENSG\d+\.\b', record.id)[0].split('.')[0]
        ENST_result = re.findall(r'\bENST\d+\.\b', record.id)[0].split('.')[0]

        AA_fa_dict[ENSG_result + '_' + ENST_result] = record.seq

with gzip.open(DNA_fa_gz_path, "rt") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        ENSG_result = re.findall(r'\bENSG\d+\.\b', record.id)[0].split('.')[0]
        ENST_result = re.findall(r'\bENST\d+\.\b', record.id)[0].split('.')[0]

        match = re.search(r"CDS:(\d+)-(\d+)", record.id)
        start_id = int(match.group(1))
        end_id = int(match.group(2))
        cds_DNA = str(record.seq)[start_id - 1:end_id]

        DNA_fa_dict[ENSG_result + '_' + ENST_result] = cds_DNA

# Single test start
kidney_beyong5 = pd.DataFrame(columns=['ensgid', 'enstid', 'AA_seq', 'DNA_seq'])
for index, row in tqdm(kidney_beyong5_id.iterrows()):
    if row['ensgid'] + '_' + row['enstid'] in AA_fa_dict.keys() and row['ensgid'] + '_' + row[
        'enstid'] in DNA_fa_dict.keys():
        AA_fa_id = row['ensgid']
        AA_fa_seq = AA_fa_dict[row['ensgid'] + '_' + row['enstid']]
        fa_id = row['enstid']
        cds_DNA = DNA_fa_dict[row['ensgid'] + '_' + row['enstid']]
    else:
        fa_id = None
        fa_seq = None

    # Extract complete DNA IDs and sequences
    if fa_id != None and cds_DNA != None:
        if len(cds_DNA) % 3 == 0:
            AA_cds_DNA = dna_to_amino_acid(cds_DNA)
            if len(AA_cds_DNA) == len(AA_fa_seq):
                if AA_cds_DNA == AA_fa_seq:
                    AA_DNA_match = 1
                else:
                    AA_DNA_match = 0
            else:
                if len(AA_cds_DNA) - 1 == len(AA_fa_seq):
                    if AA_cds_DNA[0:len(AA_cds_DNA) - 1] == AA_fa_seq:
                        AA_DNA_match = 1
                else:
                    AA_DNA_match = 0

            if AA_DNA_match == 1:
                kidney_beyong5.loc[len(kidney_beyong5)] = [row['ensgid'], row['enstid'],
                                                           str(AA_cds_DNA).strip('[').strip(']').replace("\'", ""),
                                                           str(cds_DNA).strip('[').strip(']').replace("\'", "")]

kidney_TPM_beyong5_all = kidney_beyong5
cds_kidney_TPM_beyong5_all = kidney_beyong5['DNA_seq']

high_TPM_DNA_file_fasta = './data/train_data/csd_kidney_high_TPM_200_2000.fasta'
fp = open(high_TPM_DNA_file_fasta, 'w')

for index, row in kidney_TPM_beyong5_all.iterrows():
    if 200 <= len(row['DNA_seq']) <= 2000 and len(row['DNA_seq']) % 3 == 0:
        seq_name = ">" + str(row['ensgid']) + '_' + (str)(row['enstid']) + "\n"
        seq = row['DNA_seq']
        fp.write(seq_name)
        fp.write(seq + "\n")
fp.close()

cds_kidney_TPM_beyong5_select = [x for x in cds_kidney_TPM_beyong5_all if 200 <= len(x) <= 2000 and len(x) % 3 == 0]

high_TPM_DNA_file_fasta = "./data/train_data/csd_kidney_high_TPM_200_2000.fasta"
fasta_path = high_TPM_DNA_file_fasta

# temp file
temp_file = "temp.txt"

command_line = env_path + "/bin/RNAfold -p --MEA < %s > %s" % (fasta_path, temp_file)
os.system(command_line)
mfe_dict = extract_MFE(temp_file)
os.remove(temp_file)

for name in os.listdir(sys.path[0].split('/draw_pictures')[0]):
    if name.endswith(('.ps')):
        os.remove(sys.path[0].split('/draw_pictures')[0] + '/' + name)
for name in os.listdir(sys.path[0]):
    if name.endswith(('.ps')):
        os.remove(sys.path[0] + '/' + name)

# Calculate CAI
command_line = env_path + "/bin/_cai -seqall %s -cfile Ehuman.cut -outfile %s" % (fasta_path, temp_file)
os.system(command_line)
cai_dict = extract_cai(temp_file)
os.remove(temp_file)
# Integration of calculated results
metrics_result = merge_func(['MFE', 'CAI'], mfe_dict, cai_dict)
kidney_CAI_MFE = metrics_result
DNA_fasta_path = "./data/train_data/csd_kidney_high_TPM_200_2000.fasta"
MFE200_names = kidney_CAI_MFE.loc[kidney_CAI_MFE['MFE'] < -200, 'name'].tolist()
CAI7_names = kidney_CAI_MFE.loc[kidney_CAI_MFE['CAI'] > 0.7, 'name'].tolist()
get_names = list(set(MFE200_names) & set(CAI7_names))

#  Read FASTA file and extract sequences based on selected names
sequences_names = []
sequences = []
for record in SeqIO.parse(DNA_fasta_path, "fasta"):
    if record.id in get_names:
        sequences_names.append(record.id)
        sequences.append(record.seq)

np.save('./data/train_data/csd_kidney_high_TPM_200_2000_MFE200_CAI7.npy', sequences)

# Save sequences as FASTA file
fasta_path = "./data/train_data/csd_kidney_high_TPM_200_2000_MFE200_CAI7.fasta"
fp = open(fasta_path, 'w')
for i in range(len(sequences_names)):
    seq_name = ">" + (str)(sequences_names[i]) + "\n"
    seq = str(sequences[i])
    fp.write(seq_name)
    fp.write(seq + "\n")
fp.close()
DNA_seq_dataset = sequences

# Load unoptimized fasta file
fasta_file = "./data/train_data/csd_kidney_high_TPM_200_2000_MFE200_CAI7.fasta"
Seq_name = []
Seq = []
seq_num = 0
for seqName, seq in readFa(fasta_file):
    Seq_name.append(seqName)
    Seq.append(seq)
AA_list = []
for idx in range(len(DNA_seq_dataset)):
    DNA_seq_dataset[idx] = DNA_seq_dataset[idx].replace('U', 'T')
    AA_seq = DNA_to_AA(DNA_seq_dataset[idx])
    AA_list.append(AA_seq)

# Generate a dictionary based on the Jcat concept, where one amino acid corresponds to one codon
keys = 'GAVLIPFWYCMSTNQDEKRHX'
values = 'GGCGCCGTGCTGATCCCCTTCTGGTACTGCATGAGCACCAACCAGGACGAGAAGCGCCACTAG'
Jcat_dict = {keys[i]: values[i * 3:i * 3 + 3] for i in range(len(keys))}
Jcat_opt_result = []
for AA_seq in AA_list:
    AA_to_mRNA = ''.join([Jcat_dict.get(char, char) for char in AA_seq])
    Jcat_opt_result.append(AA_to_mRNA)

# Store the entire sequence file as fasta
fasta_path = "./data/train_data/Jcat_kidney_high_TPM_200_2000.fasta"
fp = open(fasta_path, 'w')
for i in range(len(Seq_name)):
    seq_name = ">" + (str)(Seq_name[i]) + "\n"
    seq = str(Jcat_opt_result[i])
    fp.write(seq_name)
    fp.write(seq + "\n")
fp.close()

high_TPM_DNA_file_fasta = "./data/train_data/Jcat_kidney_high_TPM_200_2000.fasta"
fasta_path = high_TPM_DNA_file_fasta
# temp file
temp_file = "temp.txt"
command_line = env_path + "/bin/RNAfold -p --MEA < %s > %s" % (fasta_path, temp_file)
os.system(command_line)
mfe_dict = extract_MFE(temp_file)
os.remove(temp_file)

for name in os.listdir(sys.path[0].split('/draw_pictures')[0]):
    if name.endswith(('.ps')):
        os.remove(sys.path[0].split('/draw_pictures')[0] + '/' + name)
for name in os.listdir(sys.path[0]):
    if name.endswith(('.ps')):
        os.remove(sys.path[0] + '/' + name)

# Calculate CAI
command_line = env_path + "/bin/_cai -seqall %s -cfile Ehuman.cut -outfile %s" % (fasta_path, temp_file)
os.system(command_line)
cai_dict = extract_cai(temp_file)
os.remove(temp_file)
metrics_result = merge_func(['MFE', 'CAI'], mfe_dict, cai_dict)
kidney_CAI_MFE = metrics_result
DNA_fasta_path = "./data/train_data/Jcat_kidney_high_TPM_200_2000.fasta"

MFE200_names = kidney_CAI_MFE.loc[kidney_CAI_MFE['MFE'] < -200, 'name'].tolist()
CAI7_names = kidney_CAI_MFE.loc[kidney_CAI_MFE['CAI'] > 0.7, 'name'].tolist()
get_names = list(set(MFE200_names) & set(CAI7_names))

#  Read FASTA file and extract sequences based on selected names
Jcat_sequences_names = []
Jcat_sequences = []
for record in SeqIO.parse(DNA_fasta_path, "fasta"):
    if record.id in get_names:
        Jcat_sequences_names.append(record.id)
        Jcat_sequences.append(record.seq)

np.save('./data/train_data/Jcat_kidney_high_TPM_200_2000_MFE200_CAI7.npy', Jcat_sequences)
DNA_file = "./data/train_data/csd_kidney_high_TPM_200_2000_MFE200_CAI7.npy"
Jcat_DNA_file = "./data/train_data/Jcat_kidney_high_TPM_200_2000_MFE200_CAI7.npy"
DNA_seq_dataset = np.load(DNA_file, allow_pickle=True)
Jcat_DNA_dataset = np.load(Jcat_DNA_file, allow_pickle=True)

real_seqs = DNA_seq_dataset[0:10000]
real_seqs_len = len(real_seqs)
test_real_seqs = DNA_seq_dataset[real_seqs_len:(DNA_seq_dataset.shape[0])]
Jcat_DNA_dataset_02 = Jcat_DNA_dataset[0:int(real_seqs_len*0.2)]
Jcat_DNA_dataset_05 = Jcat_DNA_dataset[0:int(real_seqs_len*0.5)]
Jcat_DNA_dataset_1 = Jcat_DNA_dataset[0:int(real_seqs_len)]

real_Jcat_02 = list(real_seqs)+list(Jcat_DNA_dataset_02)
real_Jcat_05 = list(real_seqs)+list(Jcat_DNA_dataset_05)
real_Jcat_1 = list(real_seqs)+list(Jcat_DNA_dataset_1)

directory_path = './data/train_data'
delete_files_in_directory(directory_path)

if (os.path.exists(result_save_path) == False):
    os.makedirs(result_save_path)

np.save(result_save_path + '/kidney_train_DNA_real_seqs.npy', real_seqs)
np.save(result_save_path + '/kidney_train_DNA_real_Jcat_02.npy', real_Jcat_02)
np.save(result_save_path + '/kidney_train_DNA_real_Jcat_05.npy', real_Jcat_05)
np.save(result_save_path + '/kidney_train_DNA_real_Jcat_1.npy', real_Jcat_1)
np.save(result_save_path + '/kidney_test_DNA_real_seqs.npy', test_real_seqs)