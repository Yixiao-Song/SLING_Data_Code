import math
import numpy as np
import openai
import tqdm
from collections import defaultdict

# Functions

def get_response(prompt: str, max_tokens=150, temperature=0.7, \
                 top_p=1, n=1, logprobs=1, stop=None, echo=True):
    response = openai.Completion.create(engine="davinci",
                                        prompt=prompt,
                                        max_tokens=max_tokens,
                                        temperature=temperature,
                                        top_p=top_p,
                                        n=n,
                                        logprobs=logprobs,
                                        stop=stop,
                                        echo=echo)
    return response


def perplexity(log_probs):
    N = len(log_probs)
    return math.exp((-1/N) * np.sum(log_probs))


def evaluate_response(response, max_tokens):
    response_dict = dict(response['choices'][0])
    text = response_dict['text']

    log_probs = response_dict['logprobs']['token_logprobs'][1:]
    ppl_prompt = perplexity(log_probs)

    return {
        'prompt_ppl': ppl_prompt,
        'text': text
    }

# Eval
# Prepare the phen_para list

with open("SLING_Code/all_phenomena_paradigms.txt", 'r') as f:
    doc = f.readlines()

phen_para = defaultdict()

for line in doc:
    key, value = line.split(':')
    val_list = value.split(',')
    val_list = [i.strip('\n') for i in val_list]
    phen_para[key] = val_list

# Go through each pair ###

with open("outputs/gpt3_sling_results.txt", 'a+') as file:

    for phen,para in tqdm.tqdm(phen_para.items()):
        print(phen)
        file.write(f"{phen}\n")
        for paradigm in para:
            correct = 0
            incorrect = 0

            with open(f"SLING_Data/{phen}/{paradigm}", 'r') as f:
                doc = f.readlines()
            for line in doc:
                sentences = line.split("@@@")
                good = sentences[0]
                bad = sentences[1].strip("\n")

                response_good = get_response(good, max_tokens = 0)
                response_bad = get_response(bad, max_tokens = 0)
                good_ppl = evaluate_response(response_good, max_tokens=0)['prompt_ppl']
                bad_ppl = evaluate_response(response_bad, max_tokens=0)['prompt_ppl']
                if good_ppl < bad_ppl:
                    correct += 1
                else:
                    incorrect += 1

            assert correct + incorrect == 1000

            print(f"\t{paradigm}\t{correct/len(doc):.4f}")
            file.write(f"\t{paradigm}\t{correct/len(doc):.4f}\n")
