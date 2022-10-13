"""
  This script provides an exmaple to wrap UER-py for generation.
  Given the beginning of a text, language model generates the rest.
"""
import sys, json, glob
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

 # Read in SLING data
sling_files = glob.glob("SLING_Data/**/*.jsonl", recursive = True)

mp_dict_list = []
for sling_file in sling_files:
	dir = sling_file.split("/")
	phenomenon = dir[1]
	paradigm = dir[2].replace(".jsonl", "")
	good_sent, bad_sent = [], []

	with open(sling_file, "r") as file:
		mp_dict_list.extend([json.loads(x) for x in file.read().strip().split("\n")])

	for mp_dict in mp_dict_list:
		good_sent.append(mp_dict["sentence_good"])
		bad_sent.append(mp_dict["sentence_bad"])
	
	print(f"LOADED\tPHENOMENON {phenomenon}\tPARADIGM {paradigm}")
	with open("outputs/lstm_sling.txt", 'a+') as file:
		file.write(f"{phenomenon}\n\t{paradigm}\n")

	good_ppl= run_lstm(good_sent)
	bad_ppl = run_lstm(bad_sent)

	assert len(good_ppl) == len(bad_ppl)

	good_higher = 0
	for i,j in zip(good_ppl,bad_ppl):
		if i < j:
			good_higher += 1

	accuracy = 100*good_higher/len(good_sent)
#				 import pdb; pdb.set_trace()
#				 pass
	print('\t',accuracy)
	with open("outputs/lstm_sling.txt", 'a+') as file:
		file.write(f"\t{accuracy:.4f}\n")
