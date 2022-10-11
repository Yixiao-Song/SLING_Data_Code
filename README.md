# SLING: Sino-Linguistic Evaluation of Large Language Models

This is the official codebase accompanying our EMNLP 2022 paper "SLING: Sino-Linguistic Evaluation of Large Language Models". You can find our paper on arXiv [here](TBD).

## SLING Dataset

See [`SLING_Data`](SLING_Data) and the readme file in it.

## SLING Code

See [`SLING_Code`](SLING_Code) and the readme file in it.

## Setup

```
python -m virtualenv sling-venv
source sling-venv/bin/activate
pip install torch torchvision # currently, this is the version compatible with CUDA 10.1
pip install transformers
pip install nltk
pip install --editable .
```

## To run the language models

```
python SLING_Code/lm_sling.py
```