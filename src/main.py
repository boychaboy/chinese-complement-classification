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
    
    # Load sentences
    labels = ['下去','下来','出来','起来','上去','上来','过来','过去']
    
    if args.model == 'bert':
        model = BertForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=len(labels))
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    # train_sent, train_label = mask_data(data, labels, 500)
    train_data = json.load(open(args.train_data))
    val_data = json.load(open(args.val_data))
    args.eval_batch_size = args.train_batch_size * 4 

    train_dataloader, validation_dataloader = preprocess(train_data, val_data, labels, tokenizer, args)

    optimizer = AdamW(model.parameters(),
            lr = args.lr, 
            eps = args.eps, 
            weight_decay = args.weight_decay
            )
       
    total_steps = len(train_dataloader) * args.epochs
    scheduler = scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = total_steps * 0.1,
                                            num_training_steps = total_steps)
    model = model.to(args.device)

    args.checkpoint = total_steps / 10

    trainer = Trainer(model, optimizer, scheduler, train_dataloader, validation_dataloader, args)
    trainer.run()
