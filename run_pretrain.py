from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
import pandas as pd
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,Dataset
from torch.utils.data.distributed import DistributedSampler
import jieba
import tokenization_word as tokenization
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification,LineByLineTextDataset,BertForMaskedLM, DataCollatorForLanguageModeling
from transformers import BertTokenizer,TrainingArguments,Trainer
from transformers import AdamW
import random
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    random.seed(7580)
    np.random.seed(7580)
    torch.manual_seed(7580)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(7580)
    tokenizer = BertTokenizer('')
    bert_config = BertConfig.from_pretrained('',hidden_dropout_prob=0.1)

    # Prepare model
    model = BertForMaskedLM(bert_config).from_pretrained('')
    model.to(device)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    train_dataset= LineByLineTextDataset(tokenizer = tokenizer,file_path = '',block_size = 256)
    training_args = TrainingArguments(
    output_dir='',
    overwrite_output_dir=True,
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    save_total_limit=10,
    save_steps=10000,
    )
    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model('')


if __name__ == "__main__":
    main()
