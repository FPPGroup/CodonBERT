# CodonBERT

This is the code for the article _CodonBert: a BERT-based architecture tailored for codon optimization using the cross-attention mechanism_.


## Table of Contents

- [CodonBERT](#codonbert)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [For prediction](#for-prediction)
    - [For metrica calculation](#for-metrica-calculation)
    - [For training](#for-training)
  - [Citation](#citation)


## Installation

### Dependencies

Here is the dependencies of 
```
sklearn=1.2.1
pandas=1.5.3
numpy=1.23.5
torch=1.12.0+cu102
Biopython=1.81
tqdm=4.65.0
tensorboardX=2.6
pandas=1.4.3
emboss=6.6.0
RNAfold
```

We can install it manually by using the commands below:
```bash
conda create env -f codonbert.yaml -n Codon_Bert
conda activate Codon_Bert
git clone https://github.com/FPPGroup/CodonBERT.git
cd CodonBERT
```

## Usage
Here is the pipeline for processing, encoding data, and prediction.


### For prediction
```bash
Download the trained model
link：https://pan.baidu.com/s/1_fTWgylKz9IjP0EIzPyBgQ 
extraction code：kz65
```
```bash
python prediction.py -f <where-is-protein.fasta> -o <target-dir-to-out.fasta>
```
#### example
```bash
python prediction.py -f test_five.fasta -o result_data/test_five_result.fasta
```
### For metrica calculation
Four indicators of mRNA sequence: CAI MFE ENC GC, are calculated and stored in CSV
```bash
python get_metrics.py -e "env_path" -f 'XXX.fasta' -o "XXX.csv"
```
#### options:

```bash
  -h, --help            show this help message and exit
  -e ENV_PATH, --env_path ENV_PATH
                        environment absolute path
  -f FASTA, --fasta FASTA
                        mRNA fasta
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        metrics result path
```

#### example
```bash
python get_metrics.py -e /mnt/public2/jiangl/miniconda3/envs/CodonBERT_env -f epoch320_5_out_fix.fasta -o epoch320_5_out_fix_result.csv
```

### For training
```bash
python train.py -i <where-is-mRNA-seq.npy> 
```
#### example
```bash
python train.py -i csd_kidney_high_TPM.npy
```


## Citation


                        


