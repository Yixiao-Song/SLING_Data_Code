"""
This python script is used for running 
byt5 on the minimal pairs.
"""

########## Set up ##########

import glob, json
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import T5ForConditionalGeneration
from utils import run_masked_models,run_causal_models,AveragePerplexity,AvePplGoodBad

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
	with open("outputs/byt5_result_sling.txt", 'a+') as file:
		file.write(f"{phenomenon}\n\t{paradigm}\n")

	byt5_lm_names = ["google/byt5-small", "google/byt5-large"]

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
		with open("outputs/byt5_result_sling.txt", 'a+') as file:
			file.write(f"\t{name}\t{accuracy*100:.5f}\t{ave_pppl_good:.5f}\t{ave_pppl_bad:.5f}\n")
