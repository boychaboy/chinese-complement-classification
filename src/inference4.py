import argparse
import json
import os
import random
from transformers import BertForSequenceClassification
from transformers import BertTokenizer, BertModel, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch import softmax
import numpy as np
import torch
from torch import nn


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, \
            default="data/test.json", help='test data path')
    parser.add_argument("--model_path", type=str, \
            default="models/baseline/baseline.tar", help='model path')
    parser.add_argument("--model_type", type=str, \
            default="bert-base-chinese", help='model type')
    parser.add_argument("--result_path", type=str, \
            default="models/baseline/test_report.txt", help='output path')
    parser.add_argument("--wrong_path", type=str, \
            default="models/baseline/wrong_sent.txt", help='wrong_sent_path')
    parser.add_argument("--one_sent", action='store_true')
    args = parser.parse_args()
    
    test_path = open(args.test_path, 'r', encoding='utf-8')

    labels = ['上去','下去','下来','出来','起来']

    print(f"Loading model from '{args.model_path}'...")
    model = torch.load(args.model_path)
    model = model.to('cuda')
    model.eval()
   
    tokenizer = BertTokenizer.from_pretrained(args.model_type)
    print("Done!\n")
    
    test = open(args.test_path, 'r')
    test_data = json.load(test)
    total_num = len(test_data)
    total_right = 0

    test_w = open(args.result_path, 'w')
    wrong_w = open(args.wrong_path, 'w')
    print(f"Running inference with test data...")
    for data in test_data:
        sent, label = data.split('\t')
        
        if args.one_sent:
            if '[MASK]' in sent:
                all_sents = sent.split("。")
                for s in all_sents:
                    if '[MASK]' in s:
                        sent = s

        sent_ids = tokenizer(sent)['input_ids']
        sent_tensor = torch.tensor(sent_ids).unsqueeze(0).to('cuda')
        output = model(sent_tensor)
        output = output[0]
        logits = output.detach().cpu().numpy()[0]
        predict = labels[np.argmax(logits, axis=0)]
        if predict == label:
            total_right += 1
            test_w.write('o ')
        else : 
            test_w.write('x ')
        softmax_output = softmax(output, 1)
        test_w.write(sent + '\n')
        test_w.write(f"Answer : {label}\n")
        test_w.write(f"Predict : {predict}\n")
        for prob, lab in zip(softmax_output[0].tolist(),labels):
            test_w.write("\t"+lab+":"+str(int(prob*100))+"%"+'\n')
        test_w.write("\n")
        if predict != label : 
            wrong_w.write(sent + '\n')
            wrong_w.write(f"Answer : {label}\n")
            wrong_w.write(f"Predict : {predict}\n")
            for prob, lab in zip(softmax_output[0].tolist(),labels):
                wrong_w.write("\t"+lab+":"+str(int(prob*100))+"%"+'\n')
            wrong_w.write("\n")
    print(f"Test accuracy : {(total_right/total_num)*100:.2f}")

