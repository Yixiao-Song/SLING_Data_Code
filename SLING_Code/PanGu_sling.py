import argparse
import os
import sys
import re
import pdb
import copy
import nltk
import torch
import tqdm
import random
import argparse
import numpy as np
from nltk import Tree
from collections import defaultdict
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from utils import run_masked_models,run_causal_models,AveragePerplexity,AvePplGoodBad

parser = argparse.ArgumentParser()
parser.add_argument("--metric", type=str, default="perplexity")
parser.add_argument("--phenomenon", type=str, default="all_phenomena_paradigms.txt")
args = parser.parse_args()

# Read in paradigms.txt to create a dictionary
with open(f"SLING_Code/{args.phenomenon}", 'r') as f:
    doc = f.readlines()

filename = "outputs/pangu_sling_results.txt"

phen_para = defaultdict()
    
for line in doc:
    key, value = line.split(':')
    val_list = value.split(',')
    val_list = [i.strip('\n') for i in val_list]
    phen_para[key] = val_list

for phenomenon, paradigms in phen_para.items():
    print("#############",phenomenon,"#############")
    for paradigm in paradigms:
        print("#############",paradigm,"#############")

        ########## Read in minimal pairs ##########
        with open(f"SLING_Data/{phenomenon}/{paradigm}", 'r') as f:
            doc = f.readlines() 
            
        good_sent, bad_sent = [],[]

        for line in doc:
            line = line[:-1].split('@@@')
            good_sent.append(line[0])
            bad_sent.append(line[1])
            
        print(f"{len(good_sent)} good sentences and {len(bad_sent)} bad sentences are loaded.")

        with open(f"outputs/pangu_sling_results.txt", 'a+') as file:
            file.write(f"PanGu {paradigm}\n")
    
        causal_lm_names = ["imone/pangu_2_6B"]

        i = 1

        for name in causal_lm_names:

            print(f"*****Running {name}\t{i}/{len(causal_lm_names)}*****")

            model = AutoModelForCausalLM.from_pretrained(name,return_dict_in_generate=True,\
                                                            output_scores=True,trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(name,trust_remote_code=True)
            
            # import pdb;pdb.set_trace()
            # pass
            model.eval()
            model.cuda()

            accuracy,good_ppl,bad_ppl = run_causal_models(model,tokenizer,good_sent,bad_sent)
            ave_ppl_good,ave_ppl_bad = AvePplGoodBad(good_ppl,bad_ppl)

            i += 1
    
            print(f"\t{name}\t{accuracy:.5f}\t{ave_ppl_good:.5f}\t{ave_ppl_bad:.5f}\n")
            with open(f"outputs/pangu_sling_results.txt", 'a+') as file:
                file.write(f"\t{name}\t{accuracy*100:.5f}\t{ave_ppl_good:.5f}\t{ave_ppl_bad:.5f}\n")
