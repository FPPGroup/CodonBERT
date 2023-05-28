import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict
import re
import random
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os

wrok_tsv = pd.read_csv("/mnt/public2/jiangl/Projects/Project_RNA_deep_optim/data/raw_data/TPM/transcript_rna_tissue.tsv", sep='\t', header=0)
seq_file = pd.read_csv("/mnt/public2/jiangl/Projects/Project_RNA_deep_optim/data/raw_data/TPM/all_seq.csv")
tpm_val = wrok_tsv.loc[:, wrok_tsv.columns.str.contains('skeletal')]
tpm_val['TPM_mean'] = tpm_val.apply(lambda x: np.median(x), axis=1)
merged = pd.concat( [wrok_tsv.loc[:, ['ensgid', 'enstid']], tpm_val], axis=1 )
skeletal_beyong5_id = merged.loc[merged.TPM_mean > 5,'enstid']
skeletal_beyong5_id_list = list(skeletal_beyong5_id)

skeletal_high_TPM = seq_file[seq_file['enstid'].isin(skeletal_beyong5_id_list)]#不属于该组织的超过5的id
skeletal_high_TPM = skeletal_high_TPM.reset_index(drop = True)
# skeletal_high_TPM.to_csv("/mnt/public2/jiangl/Projects/Project_RNA_deep_optim/data/raw_data/TPM/train_data/skeletal_high_TPM.csv",index=0)################保存
cds_skeletal_high_TPM = skeletal_high_TPM.apply(lambda x: x['cds'],axis = 1 )
cds_skeletal_high_TPM = list(cds_skeletal_high_TPM)
# np.save("/mnt/public2/jiangl/Projects/Project_RNA_deep_optim/data/raw_data/TPM/train_data/csd_skeletal_high_TPM.npy",cds_skeletal_high_TPM)#############保存