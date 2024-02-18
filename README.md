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
We recommend `conda` to manage the computing environment. Here, the model training and prediction is based on Python and PyTorch. The calculation of CAI and MFE is based on EMBOSS v6.6.0 (Olson, 2002) and ViennaRNA v2.6.4 (Lorenz et al., 2011). The environment has been test on Ubuntu environment. As for MacOS, the EMBOSS and ViennaRNA can't be installed directly.

Here are dependencies:
```
conda create -n codonbert_env python=3.10 -y 
conda activate codonbert_env
conda install bioconda::emboss # not for macos-arm64
pip3 install torch --index-url https://download.pytorch.org/whl/cu118 # the users should check the version of pytorch
pip install ViennaRNA==2.6.4 biopython==1.81 einops==0.6.0 numpy==1.26.4 pandas==2.2.0 scikit-learn==1.2.1 tensorboardx==2.6 tqdm==4.65.0
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

For `predict.py`, the user only needs to ensure the paths of the `*.pt` file (model weights), the protein sequence file, and the output mRNA file.

```bash
python predict.py -m $path_to_MODEL_WEIGHTS -i $path_to_Amino_Acid_FASTA -o $path_to_output
```

The weights of four trained models were stored in `models/kidney_1_1_CodonBert_model_20230726_320_model_param.pt`. Users can test the code by the following commands: 

```bash
## test commmand line
python predict.py -m models/kidney_1_1_CodonBert_model_20230726_320_model_param.pt -f data/example_data/test_example.fasta -o data/example_data/optimized.fasta
```

Moreover, we've already integrated the CAI and MFE calculation in our repository. Users can assess the numeric metrics of optimized codon sequences.

```bash
python scripts/get_metrics.py -f data/example_data/optimized.fasta -o data/example_data/optimized_metrics.csv
```


### For developers

CodonBERT is supposed to be trained easily and flexibly. Thus, developers only nned to foucs on custom data processing. We provided two training approach.


1. ==Train on pretrained CodonBERT when the data size is small [Fine-tune].==

line XXX-XXX in XXX.py , epoch and lr is in line xxx, xxx.

2. ==Train a new model without trained weights when the data size exceeds 5k.==

```bash
python train.py -i $path_to_fastq -v  -o $path_to_save_model_weights
```

The detailed options of `train.py` is listed below:
```bash
python train.py -h
```


### Data processing in our paper

```
python ./scripts/data_preocessing.py -t -l -c -o
```
[解释说明]



## Citation
Brandes,N. et al. (2022) ProteinBERT: a universal deep-learning model of protein sequence and function. Bioinformatics, 38, 2102–2110.

Lorenz,R. et al. (2011) ViennaRNA Package 2.0. Algorithms for Molecular Biology, 6, 26.

Olson,S.A. (2002) EMBOSS opens up sequence analysis. European Molecular Biology Open Software Suite. Brief Bioinform, 3, 87–91.