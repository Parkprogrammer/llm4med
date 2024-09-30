import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
import bitsandbytes as bnb
from OpenBioLLM.qlora2 import QLoRAWrapper

class RecommenderDataset(Dataset):
    
    def __init__(self, data, tokenizer, add_token=False, new_token="[RECOMMEND]", max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.add_token = add_token
        self.new_token = new_token
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    # TODO: Need to change the Dataset item in the form of template
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.add_token:
            text = item["preferences"] + f" {self.new_token} " + item["recommendation"]
        else:
            text = item["preferences"] + " " + item["recommendation"]
        
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {key: val.squeeze(0) for key, val in encodings.items()}
    
class RecommenderTrainer:
    
    def __init__(self, model, tokenizer, data, device, qlora_params=None):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data
        self.device = device
        self.dataset = None
        self.dataloader = None
        self.qlora_wrapper = QLoRAWrapper(**(qlora_params or {})) # Don't exactly know the grammar behind this
    
    def prepare_model_for_training(self):
        self.model = self.qlora_wrapper.prepare_model(self.model)

    def prepare_dataset(self):
        self.dataset = RecommenderDataset(self.data, self.tokenizer, self.tokenizer.add_token, self.tokenizer.new_token)
        self.dataloader = DataLoader(self.dataset, batch_size=12, shuffle=True)
        # NOTE: Need to change the batch_size(It was actually set to in a large dataset 12), maybe add some more augmentations
    
    # NOTE: The num_epochs were set to 4 in the original platform
    '''
        1. Need to change the paramater into a config yaml file
    '''
    def train(self, num_epochs=4):
        
        self.prepare_model_for_training()
        
        optimizer = bnb.optim.AdamW8bit(self.model.parameters(), lr=0.0002)
        
        total_steps = len(self.dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
        # NOTE: Need more detail about the warmup_steps and adjust it
        
        # TODO : Need to add training projection layer & Output head MLP
        self.model.train()
        for epoch in range(num_epochs):
            for batch in self.dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
    def save_model(self, path):
        self.model.save_pretrained(f"{path}/model")
        self.tokenizer.save_pretrained(f"{path}/tokenizer")