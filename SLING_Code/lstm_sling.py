"""
  This script provides an exmaple to wrap UER-py for generation.
  Given the beginning of a text, language model generates the rest.
"""
import sys
import os
import argparse
import torch
import tqdm
import torch.nn.functional as F
from collections import defaultdict

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.embeddings import *
from uer.encoders import *
from uer.targets import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.model_loader import load_model
from uer.opts import infer_opts, tokenizer_opts

def run_lstm(sent_list):
    sent_ppl_list = [] 
    
    for sent in tqdm.tqdm(sent_list):
        src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(sent))
        seg = [1] * len(src)
        beginning_length = len(src)
        if len(src) > args.seq_length:
            src = src[:args.seq_length]
            seg = seg[:args.seq_length]
        src_tensor, seg_tensor = torch.LongTensor([src]), torch.LongTensor([seg])
        src_tensor = src_tensor.cuda()
        seg_tensor = seg_tensor.cuda()

        with torch.inference_mode():
            output = model(src_tensor, seg_tensor)[0]
            all_log_probs = torch.log_softmax(output, dim=-1)
            token_log_probs = torch.gather(all_log_probs[:-1, :], 1, src_tensor[0, 1:].unsqueeze(1))
            token_log_prob_list = [i.item() for i in token_log_probs]
            sent_log_prob_pair[sent] = tuple(token_log_prob_list)

            ppl = torch.exp(-1 * token_log_probs.squeeze().mean())
            sent_ppl_list.append(ppl)
    return sent_ppl_list

class GenerateLm(torch.nn.Module):
    def __init__(self, args):
        super(GenerateLm, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.target = str2target[args.target](args, len(args.tokenizer.vocab))

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = self.target.output_layer(output)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    tokenizer_opts(parser)

    args = parser.parse_args()

    args = load_hyperparam(args)
    args.target = "lm"
    args.batch_size = 1

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    model = GenerateLm(args)
    model = load_model(model, args.load_model_path)
    model.cuda()
    model.eval()

    # Read in all_phenomea_paradigms.txt
    
    with open(f"SLING_Code/all_phenomena_paradigms.txt", 'r') as f:
        doc = f.readlines()

    phen_para = defaultdict()
    
    for line in doc:
        key, value = line.split(':')
        val_list = value.split(',')
        val_list = [i.strip('\n') for i in val_list]
        phen_para[key] = val_list

    ### Go thru all sentences and calc token log prob ###
    sent_log_prob_pair = defaultdict(list)
        
    for phenonemon, paradigms in phen_para.items():
        for paradigm in paradigms:
            temp_dir = f"{phenonemon}/{paradigm}"
            

            with open(f"SLING_Data{prefix}/{temp_dir}", 'r') as file:
                doc = file.readlines()

            good_sent, bad_sent = [], []

            for line in doc:
                line = line[:-1].split('@@@')
                good_sent.append(line[0])
                bad_sent.append(line[1])

            good_ppl= run_lstm(good_sent)
            bad_ppl = run_lstm(bad_sent)

            assert len(good_ppl) == len(bad_ppl)

            good_higher = 0
            for i,j in zip(good_ppl,bad_ppl):
                if i < j:
                    good_higher += 1

            accuracy = 100*good_higher/len(good_sent)
    #                 import pdb; pdb.set_trace()
    #                 pass
            print(phenonemon,paradigm,'\t',accuracy)
            with open('outputs/lstm_results.txt', 'a+') as file:
                file.write(f"{temp_dir}\n")
                file.write(f"{accuracy:.4f}\n")
