from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import VisionEncoderDecoderConfig, AutoConfig, default_data_collator, Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torch.optim import Adam
import pickle as pkl
import os
import numpy as np
from datasets import load_metric
import torchvision
from basic.transforms import apply_data_augmentation
from basic.data import OCRDataset
from basic.metric import eval_metrics
import torch.nn as nn
import yaml
import argparse

def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

cer_metric = load_metric("cer")
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", help = "Select the file in the config folder")
args = parser.parse_args()
config = load_config('{}'.format(args.cfg))       

db_nm = config['cfg_data']['db_nm']
max_range = config['cfg_data'].get('max_range', None)
max_char = config['cfg_data']['max_char']
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

if max_range is not None: 
    dm_nms = ['{}_{:02d}'.format(db_nm, i) for i in range(max_range)]
    trainset = ConcatDataset([
        OCRDataset('train', db_nm, processor, max_char, aug=config.get('cfg_aug', None), mutate_prob=0.0, level=0)
        for db_nm in dm_nms
    ])
    validset = ConcatDataset([
        OCRDataset('valid', db_nm, processor, max_char, level=0)
        for db_nm in dm_nms
    ])
    testset  = ConcatDataset([
        OCRDataset('test' , db_nm, processor, max_char, level=0)
        for db_nm in dm_nms
    ])
else:
    trainset = CRDataset('train', db_nm, processor, max_char, aug=config.get('cfg_aug', None), mutate_prob=0.0, level=0)
    validset = OCRDataset('valid', db_nm, processor, max_char, level=0)
    testset  = OCRDataset('test' , db_nm, processor, max_char, level=0)

num_workers = config['cfg_trainer']['num_workers'] if os.name != 'nt' else 0

configuration = AutoConfig.from_pretrained("microsoft/trocr-base-handwritten")
configuration.encoder.hidden_dropout_prob = config['cfg_model']['encoder_hidden_dropout']
configuration.encoder.attention_probs_dropout_prob = config['cfg_model']['encoder_attn_dropout']

configuration.decoder.dropout = config['cfg_model']['decoder_hidden_dropout']
configuration.decoder.attention_dropout = config['cfg_model']['decoder_attn_dropout']
configuration.decoder.decoder_layerdrop = config['cfg_model']['decoder_layer_dropout']

configuration.decoder_start_token_id = processor.tokenizer.cls_token_id
configuration.pad_token_id = processor.tokenizer.pad_token_id
configuration.vocab_size = processor.tokenizer.vocab_size
configuration.eos_token_id = processor.tokenizer.eos_token_id

model = VisionEncoderDecoderModel.from_pretrained(config['cfg_trainer']['ckpt'], config = configuration)

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=config['cfg_trainer']['predict_with_generate'],
    generation_num_beams=config['cfg_trainer']['generation_num_beams'],
    generation_max_length=config['cfg_trainer']['generation_max_length'],
    evaluation_strategy=config['cfg_trainer']['evaluation_strategy'],
    per_device_train_batch_size=config['cfg_trainer']['train_bs'],
    per_device_eval_batch_size=config['cfg_trainer']['eval_bs'],
    dataloader_num_workers=num_workers,
    fp16=True,
    output_dir=config['cfg_trainer']['output_dir'],
    num_train_epochs=config['cfg_trainer']['num_train_epochs'],
    logging_steps=config['cfg_trainer']['logging_steps'],
    save_steps=config['cfg_trainer']['save_steps'],
    eval_steps=config['cfg_trainer']['eval_steps'],
    save_total_limit=2,
    metric_for_best_model=config['cfg_trainer']['metric_for_best_model'],
    greater_is_better=config['cfg_trainer']['greater_is_better'],
    
    #load_best_model_at_end=True,
    learning_rate=config['cfg_trainer']['learning_rate'],
    warmup_ratio=config['cfg_trainer']['warmup_ratio'],
    weight_decay=config['cfg_trainer']['weight_decay'],
    lr_scheduler_type = 'cosine'
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.image_processor, #image_processor
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=trainset,
    eval_dataset=validset,
    data_collator=default_data_collator,
)

trainer.train(True)
    
valid_stats = eval_metrics(model, validset, processor, batch_size=8, num_workers=num_workers, num_beam=1, max_length=450, cer_stats=True)
test_stats = eval_metrics(model, testset, processor, batch_size=8, num_workers=num_workers, num_beam=1, max_length=450, cer_stats=True)

def save_stats(stats, output_dir, nm):
    with open('{}/{}'.format(output_dir, nm), 'wb') as f:
        pkl.dump(stats, f)

save_stats(valid_stats, training_args.output_dir, 'valid_stats.pkl')
save_stats(test_stats, training_args.output_dir, 'test_stats.pkl')

print('valid_cer: {}'.format(valid_stats['cer']))
print('test_cer:  {}'.format(test_stats['cer']))
print('valid_wer: {}'.format(valid_stats['wer']))
print('test_wer:  {}'.format(test_stats['wer']))
