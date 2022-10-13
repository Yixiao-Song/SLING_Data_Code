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
	with open("outputs/pangu_sling_results.txt", 'a+') as file:
		file.write(f"{phenomenon}\n\t{paradigm}\n")

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

		print(f"\t{name}\t{accuracy*100:.5f}\t{ave_ppl_good:.5f}\t{ave_ppl_bad:.5f}\n")
		with open(f"outputs/pangu_sling_results.txt", 'a+') as file:
			file.write(f"\t{name}\t{accuracy*100:.5f}\t{ave_ppl_good:.5f}\t{ave_ppl_bad:.5f}\n")
