from rouge_score import rouge_scorer
from tqdm import tqdm
from collections import Counter
from nltk import word_tokenize
import json
from typing import List,Tuple,Dict

def post_processing(outputs):
    return [extract_summary(output) for output in outputs]


def extract_summary(sentence:str):
    sentence = sentence.strip().split("[[SEP]]")
    while len(sentence) > 0:
        summary = sentence.pop(0).strip()
        s = summary.replace("\'","").replace("\"","").replace("`","").replace("”","").replace("’ s","").strip()
        if s not in ["","Sentence:"]:
            break
    return summary


def sentences_post_processing(hyps,srcs,tgts):
    pp_hyps,pp_srcs,pp_tgts = [],[],[]
    for hyp,src,tgt in zip(hyps,srcs,tgts):
        for mark in ["``", "`", "\'\'", "\'", "\"\"", "\"","”"]:
            if mark in hyp and mark not in src:
                hyp.replace(mark,"")
        hyp = hyp.replace("``","\"")
        hyp = hyp.replace("''","\"")
        hyp = hyp.replace(" ’ s "," 's ")

        src = src.replace("``","\"")
        src = src.replace("''","\"")
        src = src.replace(" ’ s "," 's ")

        tgt = tgt.replace("``","\"")
        tgt = tgt.replace("''","\"")
        tgt = tgt.replace(" ’ s "," 's ")

        pp_hyps.append(hyp.strip())
        pp_srcs.append(src.strip())
        pp_tgts.append(tgt.strip())
    
    return pp_hyps,pp_srcs,pp_tgts



def compute_token_f1(tgt_tokens, pred_tokens, use_counts=True):
    if not use_counts:
        tgt_tokens = set(tgt_tokens)
        pred_tokens = set(pred_tokens)
    tgt_counts = Counter(tgt_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = 0
    for t in (set(tgt_tokens) | set(pred_tokens)):
        overlap += min(tgt_counts[t], pred_counts[t])
    p = overlap / len(pred_tokens) if overlap > 0 else 0.
    r = overlap / len(tgt_tokens) if overlap > 0 else 0.
    f1 = (2 * p * r) / (p + r) if min(p, r) > 0 else 0.
    return f1



def src_tgt_preprocess(text):
    text_tok = word_tokenize(text)
    text_tok = " ".join(text_tok)
    # text = text.replace("``","”")
    # text = text.replace("''","”")
    text = text.lower()
    return text_tok

def hyp_preprocess(text):
    text_tok = word_tokenize(text)
    text_tok = " ".join(text_tok)
    # text = text.replace("``","”")
    # text = text.replace("''","”")
    # text = text.replace(" ’ s "," 's ")
    text = text.lower()
    return text_tok


def get_rouge(tgt, hyp):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(tgt, hyp)
    r1 = scores["rouge1"].fmeasure
    recall_1 = scores["rouge1"].recall
    r2 = scores["rouge2"].fmeasure
    recall_2 = scores["rouge2"].recall
    rl = scores["rougeL"].fmeasure
    recall_l = scores["rougeL"].recall
    return r1, r2, rl, recall_1, recall_2, recall_l

def get_cr(tgt, hyp, src):
    tgt = tgt.split()
    hyp = hyp.split()
    src = src.split()
    tgt_cr = len(tgt)/len(src)
    hyp_cr = len(hyp)/len(src)
    return tgt_cr, hyp_cr

def get_paraphrased_number(src, hyp):
    hyp = hyp.split()
    src = src.split()
    number = 0
    for word in hyp:
        if word not in src:
            number += 1
    return number




def evaluate(tgts:List[str],srcs:List[str],hyps:List[str]):
    """
    tgts = detokenized_summaries
    srcs = detokenized_texts
    hyps = post_processed_outputs
    """

    hyps,srcs,tgts = sentences_post_processing(hyps,srcs,tgts)


    f1s=0
    r1s, r2s, rls = 0, 0, 0
    tgt_crs, hyp_crs = 0, 0
    recalls_1,recalls_2,recalls_l = 0.0,0.0,0.0
    para_nums = 0
    for src, hyp, tgt in zip(srcs, hyps, tgts):
        src_tok = src_tgt_preprocess(src)
        tgt_tok = src_tgt_preprocess(tgt)
        hyp_tok = hyp_preprocess(hyp)
        if "”" not in src:
            hyp = hyp.replace("”","")
            hyp = hyp.strip()
        f1=compute_token_f1(tgt_tok.split(), hyp_tok.split())
        f1s+=f1

        r1, r2, rl, recall_1, recall_2, recall_l = get_rouge(tgt_tok, hyp_tok)
        r1s += r1
        r2s += r2
        rls += rl
        recalls_1 += recall_1
        recalls_2 += recall_2
        recalls_l += recall_l


        tgt_cr, hyp_cr = get_cr(tgt_tok, hyp_tok, src_tok)
        tgt_crs += tgt_cr
        hyp_crs += hyp_cr

        para_num = get_paraphrased_number(src_tok, hyp_tok)
        para_nums += para_num

    f1 = f1s/len(tgts)
    r1 = r1s/len(tgts)
    r2 = r2s/len(tgts)
    rl = rls/len(tgts)
    recall_1 = recalls_1/len(tgts)
    recall_2 = recalls_2/len(tgts)
    recall_l = recalls_l/len(tgts)
    tcr = tgt_crs/len(tgts)
    hcr = hyp_crs/len(tgts)
    n_para = para_nums/len(tgts)

    r1 = r1*100
    r2 = r2*100
    rl = rl*100
    tcr = tcr*100
    hcr = hcr*100

    r = f"""
    Avg R-1 scores: {r1:.2f}
    Avg R-2 scores: {r2:.2f}
    Avg R-L scores: {rl:.2f}
    Avg F-1 scores: {f1:.2f}
    Avg tgt-cr ratio: {tcr:.2f}
    Avg hyp-cr ratio: {hcr:.2f}
    Avg number of para words: {n_para:.4f}
    """
    print("-"*40)
    print(r)
    print("-"*40)

    return r