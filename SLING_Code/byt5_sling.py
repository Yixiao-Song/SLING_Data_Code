"""
This python script is used for running 
byt5 on the minimal pairs.
"""

########## Set up ##########

import argparse
import os
import sys
import re
import pdb
import torch
import tqdm
import random
import numpy as np
from collections import defaultdict
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import T5ForConditionalGeneration
from sling_code.utils import run_masked_models,run_causal_models,AveragePerplexity,AvePplGoodBad

with open(f"SLING_Code/all_phenomena_paradigms.txt", 'r') as f:
    doc = f.readlines()

phen_para = defaultdict()
    
for line in doc:
    key, value = line.split(':')
    val_list = value.split(',')
    val_list = [i.strip('\n') for i in val_list]
    phen_para[key] = val_list

byt5_lm_names = ["google/byt5-small", "google/byt5-large"]

with open(f"outputs/byt5_sling_results.txt", 'a+') as file:
    for phen,para in tqdm.tqdm(phen_para.items()):
        print(phen)
        file.write(f"{phen}\n")
        for paradigm in para:
            print(paradigm)
            file.write(f"{paradigm}\n")
            
            with open(f"SLING_Data/{phen}/{paradigm}", 'r') as f:
                doc = f.readlines()
            for line in doc:
                line = line[:-1].split('@@@')
                good_sent.append(line[0])
                bad_sent.append(line[1])

            i = 1

            for name in byt5_lm_names:
                print(f"*****Running {name}\t{i}/{len(byt5_lm_names)}*****")

                model = T5ForConditionalGeneration.from_pretrained(name,return_dict_in_generate=True,\
                                                                   output_scores=True)
                tokenizer = AutoTokenizer.from_pretrained(name)

                model.eval()
                model.cuda()

                accuracy,good_pppl,bad_pppl = run_masked_models(model,tokenizer,good_sent,bad_sent, func_type='byt5')
                ave_pppl_good,ave_pppl_bad = AvePplGoodBad(good_pppl,bad_pppl)

                i += 1

                print(f"\t{name}\t{accuracy:.5f}\t{ave_pppl_good:.5f}\t{ave_pppl_bad:.5f}\n")
                file.write(f"\t{name}\t{accuracy*100:.5f}\t{ave_pppl_good:.5f}\t{ave_pppl_bad:.5f}\n")
