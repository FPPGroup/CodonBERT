# CodonBERT

This is the code for the article _CodonBert: a BERT-based architecture tailored for codon optimization using the cross-attention mechanism_. CodonBERT is a flexible deep-learning model for codon optimization, which is inspired by ProteinBERT (Brandes et al., 2022). We made crucial modifications to build the CodonBERT. As for architecutre, (1) the right-side network was rebuilt to match the encoder on the left-side; (2) codon tokens are now used as both keys and values in the cross-attention mechanism, while the protein sequence serves as the query. In this way, CodonBERT learns codon usage preferences and contextual combination preferences via randomly masked codon tokens. 

CodonBERT requires amino acid sequences in FASTA format as input, and predicted the optimizaed codon sequences. Four trained models based on high-TPM data (with various proporations of JCAT-optimized sequences) are provided in this repository. The users can directly use `predict.py` to conduct codon optimization. Notably, we provided the `train.py` for developers to train a cusom model on specific data. In current version, the hyperparameters of model can only be modified in the source code. The graphic user interface is under developing till Apr. 2024. In the meantime, we're processing the tissue-specific data to realize a tissue-speific optimization tool.

[figure]


## Table of Contents

- [CodonBERT](#codonbert)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [For users](#for-users)
    - [For developers](#for-developers)
  - [Citation](#citation)


## Installation
We recommand using `conda` to build your computing environment. Here, the model training and prediction is based on Python and PyTorch. The calculation of CAI and MFE is based on EMBOSS v6.6.0 (Olson, 2002) and ViennaRNA v2.6.4 (Lorenz et al., 2011). 

Here are dependencies:
```
sklearn=1.2.1
pandas=1.5.3
numpy=1.23.5
torch=1.12.0+cu102
Biopython=1.81
tqdm=4.65.0
tensorboardX=2.6
pandas=1.4.3
```

If using `conda`, users can run the line commands below:

```bash
conda env create -f codonbert_env.yaml -n codonbert_env
conda activate codonbert_env
```

Download the source code:
```bash
git clone https://github.com/FPPGroup/CodonBERT.git
cd CodonBERT
```

## Usage
The code in this repository can be used for model training, prediction.

### Codon Optimization

[explain the options]

```bash
python predict.py -m $path_to_MODEL_WEIGHTS -i $path_to_Amino_Acid_FASTA -o $path_to_output
```

Moreover, we've already integrated the CAI and MFE calculation in our repository. Users can assess the numeric metrics of optimized codon sequences.

The weights of four trained models were stored in `models/XXX`. Users can test the code by the following commands: 

```bash
python scripts/predict.py -m models/xxxx. -i test_data/amino_acid_seq.fasta -o test_data/optimized_codon_seq.fasta
```

CAI/MFE calculation
```bash
python get_metrics.py -e CodonBERT_env -f epoch320_5_out_fix.fasta -o epoch320_5_out_fix_result.csv
```


### For developers

CodonBERT is supposed to be trained easily and flexibly. Thus, developers only nned to foucs on custom data processing. We provided two training approach.

1. Train on weighted CodonBERT when the data size is small.
```bash
python train.py -m $path_to_pretrained_model_weights -i $path_to_train_data -o $path_to_save_model_weights -epoch XX -lr XX
```

2. Train a new model without trained weights when the data size exceeds 5k.
```bash
python train.py -i $path_to_train_data -o $path_to_save_model_weights
```

The detailed options of `train.py` is listed below:
```bash
python train.py -h
```

Users can test the code using the following commands:
```bash
python scripts/train.py -i test_data/amino_acid_seq.fasta -o test_data/model_save.xx  -epoch XX
```


## Citation
Brandes,N. et al. (2022) ProteinBERT: a universal deep-learning model of protein sequence and function. Bioinformatics, 38, 2102–2110.
Lorenz,R. et al. (2011) ViennaRNA Package 2.0. Algorithms for Molecular Biology, 6, 26.
Olson,S.A. (2002) EMBOSS opens up sequence analysis. European Molecular Biology Open Software Suite. Brief Bioinform, 3, 87–91.