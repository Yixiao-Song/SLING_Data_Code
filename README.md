# SLING: Sino-Linguistic Evaluation of Large Language Models

[![arxiv](https://img.shields.io/badge/arXiv-2210.11689-b31b1b.svg)](http://arxiv.org/abs/2210.11689)

This is the official codebase accompanying our EMNLP 2022 paper "SLING: Sino-Linguistic Evaluation of Large Language Models". You can find our paper on [arxiv](https://arxiv.org/abs/2210.11689).

## SLING Dataset

See [`SLING_Data`](SLING_Data) and the readme file in it.

A complete list of all phenomea and paradigms can be found in [`PhenomenonParadigmList.txt`](PhenomenonParadigmList.txt)

## SLING Code

See [`SLING_Code`](SLING_Code) and the readme file in it.

## Setup

```
python -m virtualenv sling-venv
source sling-venv/bin/activate
pip install torch torchvision # currently, this is the version compatible with CUDA 10.1
pip install transformers
pip install nltk
```

## To run the language models

```
python SLING_Code/lm_sling.py
python SLING_Code/PanGu_sling.py
python SLING_Code/gpt3_sling.py
python SLING_Code/byt5_sling.py
```

`lstm_sling.py` is included in [`SLING_Code`](SLING_Code) for reference. Details of how to run the CLUECorpusSmall LSTM language model can be found [here](https://github.com/dbiir/UER-py/wiki/Modelzoo).

## Citation Information

If you use SLING, please cite it as follows:

```
@inproceedings{sling22,
author={Yixiao Song and Kalpesh Krishna and Rajesh Bhatt and Mohit Iyyer},
booktitle = {Empirical Methods in Natural Language Processing},
Year = "2022",
Title={SLING: Sino Linguistic Evaluation of Large Language Models},
}
```
