import argparse, glob, json, random, pdb, jieba, tqdm
from decimal import setcontext
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForMaskedLM
from statistics import mean, median

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
random.seed(1130)

### JIEBA FUNCTION ###
def jieba_cnt(sentence_list):
    print("### Using JIEBA to count word types ###")
    token_list = []
    for sentence in sentence_list:
        tokens_0 = jieba.lcut(sentence, cut_all=False)
        for token in tokens_0: 
            tokens_1 = jieba.lcut(token, cut_all=False)
            token_list.extend(tokens_1)

    skip_list = ["‘", "’", "，", "。", "？", "！", "——", "$", "《", "》"]
    token_list_set = set(token_list)
    # with compound words
    type_list = [x for x in token_list_set if x not in skip_list]

    return len(type_list)

### NGRAM FUNCTION ###
def ngram_cnt(n: int, sentence_list):
    print(f"### Counting {n}-gram types ###")
    ngram_list = []
    for sentence in sentence_list:
        tokenized_sentence = tokenizer.tokenize(sentence)
        idx = 0
        while idx <= len(tokenized_sentence) - n:
            ngram = tokenized_sentence[idx:idx+n]
            ngram_list.append("".join(ngram))
            idx += 1
    ngram_set = set(ngram_list)
    
    return len(ngram_set)

### MAIN ###

parser = argparse.ArgumentParser()
parser.add_argument('--SLING', default=True, type=bool)
parser.add_argument('--CLiMP', default=True, type=bool)
args = parser.parse_args()

skip_file = ["SLING-data/MP_Classifier/mp_cl_adj_comp_noun.txt",
             "SLING-data/MP_Classifier/mp_cl_comp_noun.txt"]

if args.SLING == True:
    print("++++++ SLING ++++++")
    files = []
    for pattern in ['SLING-data/*', 'SLING-data/*/*', 'SLING-data/*/*/*']:
        files.extend([x for x in glob.glob(pattern) if x.endswith(".txt")])

    sentences = []
    for file in files:
        if file in skip_file:
            continue
        print(file)
        with open(f"{file}", "r") as f:
            doc = f.readlines()
            for line in doc:
                sent1, sent2 = line.strip().split("@@@")
                sentences.extend([sent1,sent2])
    
    random.shuffle(sentences)
    sent_length = [len(x) for x in sentences]
    ave_sent_length = mean(sent_length)
    print(f"On average, a sentence in SLING has {ave_sent_length} characters (median {median(sent_length)}).")
        
    ### 32K only to be comparable to CLiMP ###
    print(f"randomly select 32K sentences")
    for i in range(1,5):
        ngram_type_cnt = ngram_cnt(i, sentences[:32000])
        print(f"\tType count of {i}-gram: {ngram_type_cnt}")

    jieba_type_cnt = jieba_cnt(sentences[:32000])
    print(f"\tType count according to Jieba: {jieba_type_cnt}")

    ### whole 80k of SLING ###
    print(f"all 80K sentences")
    for i in range(1,5):
        ngram_type_cnt = ngram_cnt(i, sentences)
        print(f"\tType count of {i}-gram: {ngram_type_cnt}")

    jieba_type_cnt = jieba_cnt(sentences)
    print(f"\tType count according to Jieba: {jieba_type_cnt}")

if args.CLiMP == True:
    print("++++++ CLiMP ++++++")
    files = glob.glob("CLiMP-data/*1000.csv")
    sentences = []
    for file in files:
        if "ba_" in file:
            with open(f"{file}", "r") as f:
                doc = f.readlines()
                for line in doc:
                    sent = line.strip()
                    sentences.append(sent)
        else:
            with open(f"{file}", "r") as f:
                doc = f.readlines()
                for line in doc[1:]:
                    sent = line.strip().split(",")[-2]
                    sentences.append(sent)
    
    sent_length = [len(x) for x in sentences]
    ave_sent_length = mean(sent_length)
    print(f"On average, a sentence in CLiMP has {ave_sent_length} characters (median {median(sent_length)}).")
        
    for i in range(1,5):
        ngram_type_cnt = ngram_cnt(i, sentences)
        print(f"Type count of {i}-gram: {ngram_type_cnt}")

    jieba_type_cnt = jieba_cnt(sentences)
    print(f"Type count according to Jieba: {jieba_type_cnt}")
