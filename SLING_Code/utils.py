"""
This file stores the functions that calculate the 
token log probabilities and sentence perplexities 
from causal langue models and masked masked language modesl.
"""

import os
import re
import pdb
import copy
import nltk
import torch
import glob
import tqdm
import numpy as np
from nltk import Tree
from collections import defaultdict

# Load CLiMP data from txt files.
def load_climp_dataset(prefix, filename, skip_header=True, extract=lambda x: x[-2]):
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


# Functions for causal language models. Return lists.
def get_token_log_prob(model, tokenizer, sentence):
    with torch.inference_mode():
        inputs = tokenizer(sentence, return_tensors='pt')

        if torch.cuda.is_available():
            for k, v in inputs.items():
                inputs[k] = v.cuda()

        outs = model.forward(**inputs)

        all_log_probs = torch.log_softmax(outs['logits'], dim=-1)
        token_log_probs = torch.gather(all_log_probs[0, :-1], 1, inputs['input_ids'][0, 1:].unsqueeze(1))
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, 1:])

        return tokens, token_log_probs


def get_ppl(model, tokenizer, list_of_sentences):
    all_neg_ppl = []
    all_lens = []

    for sentence in tqdm.tqdm(list_of_sentences):
        _, token_log_probs = get_token_log_prob(model, tokenizer, sentence)
        # print(token_log_probs.squeeze())
        ppl = torch.exp(-1 * token_log_probs.squeeze().mean())
        all_lens.append(len(token_log_probs.squeeze()))

        all_neg_ppl.append(-1 * ppl)

    return all_neg_ppl, all_lens


def get_prob(model, tokenizer, list_of_sentences):
    all_prob = []

    for sentence in tqdm.tqdm(list_of_sentences):
        _, token_log_probs = get_token_log_prob(model, tokenizer, sentence)
        # print(token_log_probs.squeeze())
        prob = token_log_probs.squeeze().sum()

        all_prob.append(prob)

    return all_prob


# Functions for masked language models. Return lists.
def get_token_pll(model, tokenizer, sentence):  #pseudo log likelihood
    token_log_probs = []
    MASK = tokenizer.mask_token_id

    with torch.inference_mode():
        inputs = tokenizer(sentence, return_tensors='pt')
        if torch.cuda.is_available():
            for k, v in inputs.items():
                inputs[k] = v.cuda()
        # skip first ([CLS]) and last ([SEP]) tokens for for loop
        for i in range(1, len(inputs['input_ids'][-1])-1, 1):
            # store a copy of token_id at mask_index position
            true_id = inputs['input_ids'][-1][i].item()
            # replace inputs['input_ids'][0, i] with [MASK] (id: 103)
            inputs['input_ids'][-1][i] = MASK

            outs = model.forward(**inputs)
            masked_token_logits = outs['logits'][-1][i]
            log_prob = torch.log_softmax(masked_token_logits, dim=-1)
            token_log_probs.append(log_prob[true_id].item())
            # replace [MASK] with true_id
            inputs['input_ids'][-1][i] = true_id

        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, 1:-1])
        return tokens, token_log_probs


def get_t5_pll(model, tokenizer, sentence): #pseudo log likelihood
    token_log_probs = []

    with torch.inference_mode():
        for cnum, char in enumerate(sentence):
            input_sent = list(sentence)
            input_sent[cnum] = ' <extra_id_0>'
            input_sent = ''.join(input_sent)

            if cnum == 0:
                continue
                # label_sent = f'{char} <extra_id_0>'
            elif cnum == len(sentence) - 1:
                label_sent = f'<extra_id_0>{char}'
            else:
                label_sent = f'<extra_id_0>{char} <extra_id_1>'

            inputs = tokenizer(input_sent, return_tensors='pt')
            labels = tokenizer(label_sent, return_tensors='pt').input_ids
            if torch.cuda.is_available():
                for k, v in inputs.items():
                    inputs[k] = v.cuda()
                labels = labels.cuda()

            outs = model.forward(**inputs, labels=labels)
            token_log_probs.append(-1 * outs.loss.item())

        return list(sentence), token_log_probs


def get_byt5_pll(model, tokenizer, sentence): #pseudo log likelihood
    token_log_probs = []
    MASK_IDX = 258

    with torch.inference_mode():
        for cnum, char in enumerate(sentence):
            prefix = sentence[:cnum]
            suffix = sentence[cnum + 1:]
            inputs = list(prefix.encode("utf-8")) + [MASK_IDX] + list(suffix.encode("utf-8")) + [MASK_IDX - 1, 1]

            if cnum == 0:
                # ignore this case since it wasn't seen in training
                continue
                # labels = list(char.encode("utf-8")) + [MASK_IDX] + [1]
            elif cnum == len(sentence) - 1:
                labels = [MASK_IDX] + list(char.encode("utf-8")) + [1]
            else:
                labels = [MASK_IDX] + list(char.encode("utf-8")) + [MASK_IDX - 1] + [1]

            inputs = torch.LongTensor([inputs])
            labels = torch.LongTensor([labels])
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            outs = model.forward(input_ids=inputs, labels=labels)
            token_log_probs.append(-1 * outs.loss.item())

        return list(sentence), token_log_probs


def get_pppl(model, tokenizer, list_of_sentences, func_type='bert'): # Pseudo PerPLexity
    all_neg_pppl = []
    if func_type == 'bert':
        func = get_token_pll
    elif func_type == 't5':
        func = get_t5_pll
    elif func_type == 'byt5':
        func = get_byt5_pll
    all_N = []

    for sentence in tqdm.tqdm(list_of_sentences):
        _, token_log_probs = func(model, tokenizer, sentence)
        sent_pll = sum(token_log_probs)
        N = len(token_log_probs)
        pppl = torch.tensor(np.exp(-sent_pll/N))
        all_N.append(N)

        all_neg_pppl.append(-1 * pppl)

    return all_neg_pppl, all_N


def get_pprob(model, tokenizer, list_of_sentences, func_type='bert'): # Pseudo Probability
    all_pprob = []
    if func_type == 'bert':
        func = get_token_pll
    elif func_type == 't5':
        func = get_t5_pll
    elif func_type == 'byt5':
        func = get_byt5_pll

    for sentence in tqdm.tqdm(list_of_sentences):
        _, token_log_probs = func(model, tokenizer, sentence)
        sent_pprob = sum(token_log_probs)

        all_pprob.append(sent_pprob)

    return all_pprob 


def AveragePerplexity(perplexity_list):
    neg_log_prob_sum = 0
    count = 0

    for i in perplexity_list:
        neg_log_prob = torch.log(-i)
        neg_log_prob_sum += neg_log_prob
        count += 1

    ave_neg_log_prob = neg_log_prob_sum/count
    ave_ppl = torch.exp(ave_neg_log_prob)

    return -ave_ppl


def AvePplGoodBad(good_ppl,bad_ppl):
    ave_ppl_good = AveragePerplexity(good_ppl)
    ave_ppl_bad = AveragePerplexity(bad_ppl)
#     print(f"Average (pseudo) perplexity for good sentences: {ave_ppl_good:.4f}")
#     print(f"Average (pseudo) perplexity for bad sentences: {ave_ppl_bad:.4f}")
    return ave_ppl_good,ave_ppl_bad

# This function finds the pairs which the models assign 
# higher probability to the bad sentence.


def find_failed_cases(good_sent_ppl, bad_sent_ppl):
    i = 0
    failed_case_idx = []

    for x,y in tqdm.tqdm(zip(good_sent_ppl, bad_sent_ppl)):
        if x < y: 
            i += 1
            failed_case_idx.append(good_sent_ppl.index(x))

    return failed_case_idx


def run_causal_models(model,tokenizer,good_sent_list,bad_sent_list, metric="perplexity"):
    if metric == "perplexity":
        good_sent_score, good_lens = get_ppl(model, tokenizer, good_sent_list)
        bad_sent_score, bad_lens = get_ppl(model, tokenizer, bad_sent_list)

    elif metric == "probability":
        good_sent_score = get_prob(model, tokenizer, good_sent_list)
        bad_sent_score = get_prob(model, tokenizer, bad_sent_list)      
    failed_case_idx = find_failed_cases(good_sent_score, bad_sent_score)
    accuracy = 1-len(failed_case_idx)/len(good_sent_score)

    return accuracy, good_sent_score, bad_sent_score


def run_masked_models(model, tokenizer, good_sent_list, bad_sent_list, func_type = 'bert', metric="perplexity"):
    if metric == "perplexity":
        good_sent_pscore, good_lens = get_pppl(model, tokenizer, good_sent_list, func_type)
        bad_sent_pscore, bad_lens = get_pppl(model, tokenizer, bad_sent_list, func_type)
    elif metric == "probability":
        good_sent_pscore = get_pprob(model, tokenizer, good_sent_list, func_type)
        bad_sent_pscore = get_pprob(model, tokenizer, bad_sent_list, func_type)
    failed_case_idx = find_failed_cases(good_sent_pscore, bad_sent_pscore)
    accuracy = 1-len(failed_case_idx)/len(good_sent_pscore)

    return accuracy, good_sent_pscore, bad_sent_pscore


# This function reads in the separated Chinese TreeBank sentences/trees
def read_trees(file):
    # Read in the file
    with open(file, 'r') as f:
        doc = f.read()

    # Split the string by '@@@'
    trees_str = doc.split('@@@')

    # To store the split trees
    trees = []

    for i in tqdm.tqdm(range(len(trees_str)-1)): # The last element is an empty string
        # Form trees from strings
        trees.append(Tree.fromstring(trees_str[i]))
    print("Read in {} trees.".format(len(trees)))
    return trees


# This function finds labels in trees
def cat_search(label_str, sentence_list):
    output = []

    for tree in sentence_list:
        labels = [x.label() for x in tree.subtrees()]

        if label_str in labels:
            output.append(tree)

    print("{} trees contain at least one {}.".format(len(output), label_str))

    return output


def male_female_count(sent_list):
    female, male = 0, 0
    for i in sent_list:
        if i[0][-3] == "她": female += 1
        else: male += 1

    print(len(sent_list), male, female)


def anaphor_template_fill_in(template_list,A,B):
    out_sent = copy.deepcopy(template_list)
    for i in range(len(template_list)):
        out_sent[i] = out_sent[i].replace("A",A)
        out_sent[i] = out_sent[i].replace("B",B)
        if out_sent[i][-1] == '\n':
            out_sent[i] = out_sent[i][:-1]
    return out_sent


def binding_template_fill_in(template_list,A,B,C):
    out_sent = copy.deepcopy(template_list)
    for i in range(len(template_list)):
        out_sent[i] = out_sent[i].replace("A",A)
        out_sent[i] = out_sent[i].replace("B",B)
        out_sent[i] = out_sent[i].replace("C",C)
        if out_sent[i][-1] == '\n':
            out_sent[i] = out_sent[i][:-1]
    return out_sent


def success_fail_apart(good_sent_list, bad_sent_list, idx):
    fail, success = [], []
    for i in range(1000):
        if i in idx:
            fail.append((good_sent_list[i],bad_sent_list[i]))
        else:
            success.append((good_sent_list[i],bad_sent_list[i]))
    return success, fail
