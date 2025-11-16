import torch
from torch.utils.data import Dataset
from basic.transforms import apply_data_augmentation
import pickle as pkl
import os
from PIL import Image
import random
import numpy as np

class OCRDataset(Dataset):
    def __init__(self, db_set, root_url, processor, max_target_length, gt_type, paragraph_by_lines_mode=False, **kwargs):
        self.root_url = root_url
        self.processor = processor
        self.max_target_length = max_target_length

        self.aug = kwargs['aug'] if 'aug' in kwargs else None
        self.mutate_prob = kwargs['mutate_prob'] if 'mutate_prob' in kwargs else None
            
            
        with open(self.root_url+'/labels.pkl','rb') as f:
            labels = pkl.load(f)
        
        self.db_set = db_set
        self.paragraph_by_lines_mode = paragraph_by_lines_mode
        self.images_url = [key for key in labels[gt_type][db_set]]
        
        if self.paragraph_by_lines_mode:
            num_lines_in_paragraph = []
            self.labels = []
            for key in self.images_url:
                lines = labels[gt_type][db_set][key]['lines']
                num_lines_in_paragraph.append(len(lines))
                self.labels += [line['text'] for line in lines]
                
            temp = [[im_url]*n_times for im_url, n_times in zip(self.images_url, num_lines_in_paragraph)]
            
            self.images_url = []
            self.line_num = []
            for tmp in temp:
                self.images_url += tmp
                self.line_num += list(range(len(tmp)))
                
        else:
            self.labels = [labels[gt_type][db_set][key]['text'] for key in labels[gt_type][db_set]]
        
    def __len__(self):
        return len(self.images_url)
    
    def ignore_pad(self, labels):
        return [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

    def mutate_lbl(self, labels, proba=0.05):
        special_tokens_and_linebreak = self.processor.tokenizer.all_special_ids+[50118]
        for i in range(self.max_target_length):
            lbl = labels[i]
            if lbl == self.processor.tokenizer.eos_token_id:
                break
            
            if random.random() < proba:
                new_lbl = random.randint(3, self.processor.tokenizer.vocab_size-1)

                while new_lbl in special_tokens_and_linebreak:
                    new_lbl = random.randint(3, self.processor.tokenizer.vocab_size-1)
                labels[i] = new_lbl
        return labels
    
    def __getitem__(self, idx):
        url = self.images_url[idx]
        path = os.path.join(self.root_url,self.db_set, url)

        #image processing
        im = Image.open(path).convert("RGB")
        lbl = self.labels[idx]
        if self.db_set == 'train':
            im, lbl = apply_data_augmentation(im, lbl, self.aug)
            
        pixel_values = self.processor(im, return_tensors="pt").pixel_values

        #lbl processing
        labels = self.processor.tokenizer(lbl, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        
        # important: make sure that PAD tokens are ignored by the loss function
        labels = self.ignore_pad(labels)
        if self.mutate_prob is not None:
            labels = self.mutate_lbl(labels, self.mutate_prob)
        
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels), 'path': path}
        if self.paragraph_by_lines_mode:
            encoding['line_num'] = torch.tensor(self.line_num[idx])
        return encoding
