import torch, pdb, argparse
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

filename = "outputs/results_sling.txt"

phen_para = defaultdict()

for line in doc:
    key, value = line.split(':')
    val_list = value.split(',')
    val_list = [i.strip('\n') for i in val_list]
    phen_para[key] = val_list

for phenomenon, paradigms in phen_para.items():
	print(f"############# {phenomenon} #############")
	with open(filename, 'a+') as file:
		file.write(f"{phenomenon}\n")

	for paradigm in paradigms:
		print(f"############# {paradigm} #############")
		with open(filename, 'a+') as file:
			file.write(f"{paradigm}\n")
		########## Read in minimal pairs ##########
		with open(f"SLING_Data/{phenomenon}/{paradigm}", 'r') as f:
			doc = f.readlines()

		good_sent, bad_sent = [],[]
		for line in doc:
			line = line.strip().split('@@@')
			good_sent.append(line[0])
			bad_sent.append(line[1])
		print(f"{len(good_sent)} good sentences and {len(bad_sent)} bad sentences are loaded.")

########## LMs ##########
##############
# Masked LMs #
##############

		masked_lm_names = ["xlm-roberta-base", "xlm-roberta-large", "bert-base-chinese", \
							"bert-base-multilingual-cased", "hfl/chinese-pert-base", \
							"hfl/chinese-pert-large", "Langboat/mengzi-bert-base", \
							"Langboat/mengzi-bert-base-fin", "nghuyong/ernie-1.0-base-zh", \
							"google/mt5-small", "google/mt5-large"]

		i = 1

		for name in masked_lm_names:

			print(f"*****Running {name}\t{i}/{len(masked_lm_names)}*****")

			if name == "google/mt5-small" or name == "google/mt5-large":
				model = MT5ForConditionalGeneration.from_pretrained(name)
				tokenizer = T5Tokenizer.from_pretrained(name)
				model.eval()
				model.cuda()
				accuracy, good_pppl, bad_pppl = run_masked_models(model,tokenizer,\
																	good_sent,bad_sent, \
																	func_type='t5',
																	metric=args.metric)
			else:
				model = AutoModelForMaskedLM.from_pretrained(name,return_dict_in_generate=True,\
																output_scores=True)
				tokenizer = AutoTokenizer.from_pretrained(name)
				model.eval()
				model.cuda()
				accuracy,good_pppl,bad_pppl = run_masked_models(model,tokenizer,good_sent,bad_sent,metric=args.metric)

			if args.metric == "perplexity":
				ave_ppl_good,ave_ppl_bad = AvePplGoodBad(good_pppl,bad_pppl)
			else:
				ave_ppl_good,ave_ppl_bad = 0.0, 0.0

			i += 1
			
			print(f"\t{name}\t{accuracy:.5f}\t{ave_ppl_good:.5f}\t{ave_ppl_bad:.5f}\n")

			# Open .txt to store the results
			with open(filename, 'a+') as file:
				file.write(f"\t{name}\t{accuracy*100:.5f}\t{ave_ppl_good:.5f}\t{ave_ppl_bad:.5f}\n")
				
	##############
	# Causal LMs #
	##############

		causal_lm_names = ["uer/gpt2-chinese-cluecorpussmall", "TsinghuaAI/CPM-Generate"]

		i = 1

		for name in causal_lm_names:

			print(f"*****Running {name}\t{i}/{len(causal_lm_names)}*****")

			model = AutoModelForCausalLM.from_pretrained(name,return_dict_in_generate=True,\
															output_scores=True)
			tokenizer = AutoTokenizer.from_pretrained(name)

			model.eval()
			model.cuda()

			accuracy,good_ppl,bad_ppl = run_causal_models(model,tokenizer,good_sent,bad_sent,metric=args.metric)
			
			if args.metric == "perplexity":
				ave_ppl_good,ave_ppl_bad = AvePplGoodBad(good_ppl,bad_ppl)
			else:
				ave_ppl_good,ave_ppl_bad = 0.0, 0.0

			i += 1
		
			print(f"\t{name}\t{accuracy:.5f}\t{ave_ppl_good:.5f}\t{ave_ppl_bad:.5f}\n")
			with open(filename, 'a+') as file:
				file.write(f"\t{name}\t{accuracy*100:.5f}\t{ave_ppl_good:.5f}\t{ave_ppl_bad:.5f}\n")
