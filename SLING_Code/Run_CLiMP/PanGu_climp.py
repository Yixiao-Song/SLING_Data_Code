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
import numpy as np
from nltk import Tree
from collections import defaultdict
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from sling_code.utils import run_masked_models,run_causal_models,AveragePerplexity,AvePplGoodBad

def load_climp_dataset(filename, skip_header=True, extract=lambda x: x[-2]):
    with open(f'CLiMP-data/{filename}', 'r') as f:
        dataset = f.readlines()

    if skip_header:
        dataset = dataset[1:]

    dataset = [x.split(',') for x in dataset]

    # confirm all elements of dataset have same length
    assert all([len(x) == len(dataset[0]) for x in dataset])
    assert len(dataset) % 2 == 0

    good_sent = [extract(dataset[i]) for i in range(0, len(dataset), 2)]
    bad_sent = [extract(dataset[i + 1]) for i in range(0, len(dataset), 2)]

    return good_sent, bad_sent

temp_names = ["anaphor_agreement_gender_1000.csv",\
              "binding_gender_1000.csv","classifier_1000.csv",\
              "classifier_adj_1000.csv","classifier_clause_1000.csv",\
              "coverb_instrument_1000.csv","coverb_with_1000.csv",\
              "filler_gap_dependency_1000.csv","head_final_clause_1000.csv",\
              "passive_formal_1000.csv","verb_complement_direction_1000.csv",\
              "verb_complement_duration_1000.csv","verb_complement_frequency_1000.csv",\
              "verb_complement_res_adj_1000.csv","verb_complement_res_verb_1000.csv"]

# temp_names = ["ba_construction_1000.csv"]

"""
When evaluating the minimal pairs, the default setting of 
load_climp_dataset can be used. However, 'ba_construction_1000.csv' 
needs to be run separately using the parameters set as 

load_climp_dataset('CLiMP-data', paradigm, skip_header=False, extract=lambda x: x[-1]) 

because of the file format. 
"""

for temp in temp_names:
    print("#############",temp,"#############")

    ########## Read in minimal pairs ##########
    good_sent, bad_sent = load_climp_dataset(temp,skip_header=True,extract=lambda x: x[-2])
   
    print(f"{len(good_sent)} good sentences and {len(bad_sent)} bad sentences are loaded.")

    with open(f"outputs/pangu_climp_results.txt", 'a+') as file:
        file.write(f"{temp}\n")
    
        causal_lm_names = ["imone/pangu_2_6B"]

        i = 1

        for name in causal_lm_names:

            print(f"*****Running {name}\t{i}/{len(causal_lm_names)}*****")

            model = AutoModelForCausalLM.from_pretrained(name,return_dict_in_generate=True,\
                                                            output_scores=True,trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(name,trust_remote_code=True)
            
            model.eval()
            model.cuda()

            accuracy, good_ppl, bad_ppl = run_causal_models(model,tokenizer,good_sent,bad_sent)
            ave_ppl_good, ave_ppl_bad = AvePplGoodBad(good_ppl,bad_ppl)

            i += 1
    
            print(f"\t{name}\t{accuracy:.5f}\t{ave_ppl_good:.5f}\t{ave_ppl_bad:.5f}\n")
            file.write(f"\t{name}\t{accuracy*100:.5f}\t{ave_ppl_good:.5f}\t{ave_ppl_bad:.5f}\n")
            