import argparse
import json
import os
import random
from transformers import BertForSequenceClassification
from transformers import BertTokenizer, BertModel, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

import numpy as np
import torch
from torch import nn
from utils import get_args, mask_data, preprocess, load_data
from trainer import Trainer
import logging

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

if __name__ == "__main__":
    args = get_args()
    logger.info(f"args:  {json.dumps(args.__dict__, indent=2, sort_keys=True)}")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    classes = dict()
    classes['过来'] = ["verb_guolai_literature.txt", "verb_guolai_media.txt", "verb_guolai_textbook.txt"]
    classes['过去'] = ["verb_guoqu_literature.txt", "verb_guoqu_media.txt", "verb_guoqu_textbook.txt"]
    classes['起来'] = ["verb_qilai_literature.txt", "verb_qilai_media.txt", "verb_qilai_textbook.txt"]
    classes['上来'] = ["verb_shanglai_literature.txt", "verb_shanglai_media.txt", "verb_shanglai_textbook.txt"]
    classes['下来'] = ["verb_xialai_literature.txt", "verb_xialai_media.txt", "verb_xialai_textbook.txt"]
    classes['下去'] = ["verb_xiaqu_literature.txt", "verb_xiaqu_media.txt", "verb_xiaqu_textbook.txt"]
    
    # Load sentences
    data = load_data(args, classes)
    labels = ['过来','过去','起来','上来','下来','下去']
    
    train_sent, train_label = mask_data(data, labels, 500)

    if args.model == 'bert':
        MODEL_TYPE = 'bert-base-chinese'
        model = BertForSequenceClassification.from_pretrained(MODEL_TYPE, num_labels=len(labels))
        tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)

    train_dataloader, validation_dataloader = preprocess(train_sent, train_label, labels, tokenizer, args)
    

    optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8 
                )
    total_steps = len(train_dataloader) * args.epochs
    scheduler = scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

    trainer = Trainer(model, optimizer, scheduler, train_dataloader, validation_dataloader, args)
    trainer.run()
