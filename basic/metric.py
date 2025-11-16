from torch.utils.data import DataLoader
from datasets import load_metric
import torch
from tqdm import tqdm

def dataset_CER(model, dataset, processor, **kwargs): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, kwargs['batch_size'], kwargs['num_workers'])
    cer_metric = load_metric("cer")
    path_str = []
    pred_str = []
    label_str = []
    
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            pixel_values, labels_ids, im_path = batch['pixel_values'], batch['labels'], batch['path']
            pred_ids = model.generate(pixel_values.to(device), num_beams=kwargs['num_beam'], max_length=kwargs['max_length'], early_stopping=True if kwargs['num_beam'] > 1 else False)
            pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
            pred_str.extend(processor.batch_decode(pred_ids, 
                                               skip_special_tokens=True))
            
            labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
            label_str.extend(processor.batch_decode(labels_ids,
                                                    skip_special_tokens=True))
            path_str.extend(im_path)
    total_cer = cer_metric.compute(predictions=pred_str, references=label_str)
    cer_stats = {'cer': total_cer}
    
    if kwargs['cer_stats']:
        cer_stats = {'raw_data' : {}, 'cer': total_cer}
        for prd, lbl, path in zip(pred_str, label_str, path_str):
            cer = cer_metric.compute(predictions=[prd], references=[lbl])
            stats = {"cer": cer, "prd": prd, "lbl": lbl}
            cer_stats['raw_data'][path]=stats
            
    return cer_stats
