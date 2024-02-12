# CodonBERT

This is the code for the article _CodonBert: a BERT-based architecture tailored for codon optimization using the cross-attention mechanism_.


## Table of Contents

- [CodonBERT](#codonbert)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [For prediction](#for-prediction)
      - [options:](#options)
      - [example](#example)
    - [For metrica calculation](#for-metrica-calculation)
      - [options:](#options-1)
      - [example](#example-1)
    - [For training](#for-training)
      - [options:](#options-2)
      - [example](#example-2)
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
python prediction.py -m <where-is-model_param.pkl> -f <where-is-protein.fasta> -o <target-dir-to-out.fasta>
```
#### options:

```bash
  -h, --help            show this help message and exit
  -m Model_param_PATH, --model PKL
                        model PKL file path
  -f FASTA, --fasta FASTA
                        Amino acid fasta path
  -o OUTPUT_PATH, --output_path FASTA
                        The save file path
```
#### example
```bash
python prediction.py -m kidney_1_1_CodonBert_model_param.pkl -f test_five.fasta -o test_five_result.fasta
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
                        environment path
  -f FASTA, --fasta FASTA
                        Codon sequence fasta path
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        metrics result path
```

#### example
```bash
python get_metrics.py -e CodonBERT_env -f test_codon_sequences.fasta -o test_codon_sequences.csv
```

### For training
```bash
python train.py -t <where-is-codon-seq-of-the-training-set.npy> -v <where-is-codon-seq-of-the-validation-set.npy>
```
#### options:

```bash
  -h, --help            show this help message and exit
  -t NPY , --input  NPY
                        The codon sequence file path of the training set
  -v NPY , --input  NPY
                        The codon sequence file path of the validation set
```
#### example
```bash
python train.py -t train_codon_sequences.npy -v validation_codon_sequences.npy
```

### For data preprocessing
```bash
python Data_preprocessing.py -e "env_path" -t "transcript_rna_tissue.tsv_path" -l "gencode.v43.pc_translations.fa.gz_path" -c "gencode.v43.pc_transcripts.fa.gz_path" -o "output_path"
```
#### options:

```bash
   -h, --help            show this help message and exit

  -e ENV_PATH, --env_path 

                        environment path

  -t TSV, --tsv_path TSV

                        transcript_rna_tissue.tsv file path

  -l TRANSLATIONS_PATH, --lations 

                        gencode.v43.pc_translations.fa.gz file path

  -c TRANSCRIPTS_PATH, --scripts 

                        gencode.v43.pc_transcripts.fa.gz path

  -o OUTPUT_PATH, --output_path 

                        Result save path
```
#### example
```bash
python Data_preprocessing.py -e CodonBert_env -t transcript_rna_tissue.tsv -l gencode.v43.pc_translations.fa.gz -c gencode.v43.pc_transcripts.fa.gz -o ./data/result
```


## Citation


                        


