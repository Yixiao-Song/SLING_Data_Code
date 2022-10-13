import argparse, glob, json
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from utils import run_masked_models,run_causal_models,AveragePerplexity,AvePplGoodBad

parser = argparse.ArgumentParser()
parser.add_argument("--metric", type=str, default="perplexity")
args = parser.parse_args()

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
	with open("outputs/result_sling.txt", 'a+') as file:
		file.write(f"{phenomenon}\n\t{paradigm}\n")

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
															  good_sent,bad_sent,\
															  func_type='t5',\
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
		
		print(f"\t{name}\t{accuracy*100:.5f}\t{ave_ppl_good:.5f}\t{ave_ppl_bad:.5f}\n")

		# Open .txt to store the results
		with open("outputs/result_sling.txt", 'a+') as file:
			file.write(f"\t\t{name}\t{accuracy*100:.5f}\t{ave_ppl_good:.5f}\t{ave_ppl_bad:.5f}\n")
			
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
	
		print(f"\t{name}\t{accuracy*100:.5f}\t{ave_ppl_good:.5f}\t{ave_ppl_bad:.5f}\n")
		with open("outputs/result_sling.txt", 'a+') as file:
			file.write(f"\t\t{name}\t{accuracy*100:.5f}\t{ave_ppl_good:.5f}\t{ave_ppl_bad:.5f}\n")
