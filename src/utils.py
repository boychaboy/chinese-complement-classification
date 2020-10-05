import argparse
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def load_file(data_dir, filenames):
    sentences = []
    for filename in filenames:
        with open(data_dir + filename, 'r') as f:
            for line in f.readlines():
                sentences.append(line.strip().split('\t')[-1])
    return sentences, len(sentences)

def load_data(args, classes):
    classes_data = dict()
    for item in classes.items():
        classes_data[item[0]] = load_file(args.data_dir,item[1]) 

    return classes_data

def get_args():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--train_data", type=str, default="data/train.json")
    parser.add_argument("--val_data", type=str, default="data/val.json")
    # model
    parser.add_argument("--model", choices=['bert'], default='bert', type=str)
    # training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--checkpoint", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--adam_betas", type=str, default="(0.9, 0.98)")
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--max_grad_norm", type=float, default=0)
    parser.add_argument("--save_dir", type=str, default="models/baseline/baseline.tar")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-chinese")

    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    return args

def mask_data(data, labels, max_len):
    total_sentences = []
    total_labels = []
    for label in labels:
        s = []
        l = []
        for sent in data[label][0]:
            if label not in sent:
                continue
            sent = sent[:max_len]
            sent = sent.replace(label, '[MASK]')
            s.append(sent)
            l.append(label)
        total_sentences.extend(s)
        total_labels.extend(l)
    return total_sentences, total_labels

def preprocess(train_data, val_data, labels, tokenizer, args):
    train_sent = [data.split('\t')[0] for data in train_data]
    train_label = [data.split('\t')[1] for data in train_data]
    val_sent = [data.split('\t')[0] for data in val_data]
    val_label = [data.split('\t')[1] for data in val_data]

    tokenized_sent = [tokenizer.tokenize(sent) for sent in train_sent]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_sent]
    train_inputs = pad_sequences(input_ids, maxlen=args.max_seq_length, \
            dtype="long", truncating="post", padding="post")
    train_labels = [labels.index(x) for x in train_label]
    train_masks = []
    for seq in train_inputs:
        seq_mask = [float(i>0) for i in seq]
        train_masks.append(seq_mask)

    tokenized_sent = [tokenizer.tokenize(sent) for sent in val_sent]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_sent]
    validation_inputs = pad_sequences(input_ids, maxlen=args.max_seq_length, \
            dtype="long", truncating="post", padding="post")
    validation_labels = [labels.index(x) for x in val_label]
    validation_masks = []
    for seq in validation_inputs:
        seq_mask = [float(i>0) for i in seq]
        validation_masks.append(seq_mask)
    
    # 데이터를 파이토치의 텐서로 변환
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)

    # 배치 사이즈
    # 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
    # 학습시 배치 사이즈 만큼 데이터를 가져옴
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=args.eval_batch_size)

    return train_dataloader, validation_dataloader
