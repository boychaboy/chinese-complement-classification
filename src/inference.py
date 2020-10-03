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
            default="data/test.txt", help='test data path')
    parser.add_argument("--model_path", type=str, \
            default="save/second_save", help='model path')
    parser.add_argument("--result_path", type=str, \
            default="data/test_result.txt", help='output path')
    args = parser.parse_args()
    
    test_path = open(args.test_path, 'r')

    labels = ['过来','过去','起来','上来','下来','下去']
    MODEL_TYPE = 'bert-base-chinese'
    

    # checkpoint = torch.load(args.model_path)
    
    # model = BertForSequenceClassification.from_pretrained(MODEL_TYPE, num_labels=len(labels))
    # state_dict = torch.load(args.model_path)['model_state_dict']
    # model.load_state_dict(state_dict=state_dict)
    
    model = torch.load(args.model_path)
    model = model.to('cuda')
    model.eval()
   
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    for sent in test_path.readlines():
        sent_ids = tokenizer(sent)['input_ids']
        sent_tensor = torch.tensor(sent_ids).unsqueeze(0).to('cuda')
        output = model(sent_tensor)
        output = output[0]
        softmax_output = softmax(output, 1)
        print(sent)
        for prob, lab in zip(softmax_output[0].tolist(),labels):
            print(lab,":",str(int(prob*100))+"%")
        print("")
