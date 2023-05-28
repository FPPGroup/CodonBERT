# ProteinBert_model

## 一、数据预处理
数据收集：
```
带有mRNA在不同组织中表达量的数据
https://www.proteinatlas.org/download/transcript_rna_tissue.tsv.zip
```

```
和proteinatlas中mRNA（enstid） 对应的序列数据
https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.pc_translations.fa.gz
```
数据筛选：
```
整理，截取后的mRNA序列名所对应的序列数据
all_seq.csv mRNA序列名（enstid），氨基酸序列（20-500），mRNA序列（cds）
```
### Requirement
```
numpy
pandas
```

### usage
```
筛选高表达数据
python select_high_TPM_data.py
```

## 二、数据编码
功能：cds_seq_to_input
### Requirement
```
numpy
pandas
```

### usage
```
python proteinbert_torch_input_encode.py
py123:cds_file_path
py180~185:output_file_path
```

## 三、模型训练
### Requirement
```
protein-bert-pytorch
sklearn=1.2.1
pandas=1.5.3
numpy=1.23.5
torch=1.12.0+cu102
Biopython=1.81
tqdm=4.65.0
tensorboardX=2.6
```
### usage
```
python proteinbert_torch_pretain.py
py218~225:输入数据
py296~298:SummaryWriter设置
py316:save model path
```

## 四、模型测试
### usage
```
#模型结果还原成氨基酸序列和mRNA序列 and fix错误预测位点
python torch_model_eval.py
py165:model路径
py258:预测结果保存路径
py368:fix结果保存路径

```

## 五、mRNA序列指标计算
功能：计算优化后序列的四个指标：CAI MFE ENC GC，并存储到CSV中
### Requirement

```
pandas=1.4.3
emboss=6.6.0
RNAfold
```

### usage

```
python get_metrics.py -f 'XXX.fasta' -o "output_path/"
```

### options:

```
  -h, --help            show this help message and exit
  -f FASTA, --fasta FASTA
                        the seq file
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        the save path
```


                        


