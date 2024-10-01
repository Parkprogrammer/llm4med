import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
import bitsandbytes as bnb
from .qlora import QLoRAWrapper
import torch.nn.functional as F
from .template import STAGE_1_TEMPLATE, STAGE_2_TEMPLATE, STAGE_3_TEMPLATE
import os
import pandas as pd

os.environ['HF_HOME'] = '/data/transformer_cache'

# Function for error-handling non-strings
def safe_get(item, key):
    value = item.get(key, "")
    if pd.isna(value):
        return ""
    return str(value)

class RecommenderDataset(Dataset):
    
    def __init__(self, data, tokenizer, recommendation_embeddings, add_token=False, new_token="[RECOMMEND]", max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.recommendation_embeddings = recommendation_embeddings
        self.add_token = add_token
        self.new_token = new_token
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    # TODO: Need to change the Dataset item in the form of template
    def __getitem__(self, idx):
        try:
            item = self.data.iloc[idx] 
        except KeyError:
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
        
        # Disease Classification (combine Primary, Secondary, Tertiary into one string)
        disease_classification = ", ".join([
            safe_get(item, "Disease Classification - Primary"),
            safe_get(item, "Disease Classification - Secondary"),
            safe_get(item, "Disease Classification - Tertiary")
        ])

        # Departments (combine Main and Sub departments)
        departments = ", ".join([
            safe_get(item, "Department - Main"),
            safe_get(item, "Department - Sub")
        ])

        # Systems (combine all System columns)
        systems = ", ".join([
            safe_get(item, "System"),
            safe_get(item, "System.1"),
            safe_get(item, "System.2"),
            safe_get(item, "System.3"),
            safe_get(item, "System.4")
        ])

        # Symptoms/Complications (combine all Disease Name columns)
        symptoms_complications = ", ".join([
            safe_get(item, "Disease Name"),
            safe_get(item, "Disease Name.1"),
            safe_get(item, "Disease Name.2"),
            safe_get(item, "Disease Name.3"),
            safe_get(item, "Disease Name.4")
        ])

        # General Characteristics
        gender = safe_get(item, "Gender")
        age = safe_get(item, "Age")
        bmi = safe_get(item, "BMI")

        # Stage 1: Patient Analysis
        stage_1_input = STAGE_1_TEMPLATE.format(
            disease_classification=disease_classification,
            departments=departments,
            systems=systems,
            symptoms_complications=symptoms_complications,
            gender=gender,
            age=age,
            bmi=bmi
        )

        # process this through the model to get stage_1_output
        stage_1_output = "STAGE_1_OUTPUT_PLACEHOLDER"

        # Stage 2: Management Plan
        stage_2_input = f"{stage_1_input}\n\n{stage_1_output}\n\n{STAGE_2_TEMPLATE}"

        # process this through the model to get stage_2_output
        stage_2_output = "STAGE_2_OUTPUT_PLACEHOLDER"

        content_list = []
        for i in range(idx, min(idx + 5, len(self.data))):
            # i번째 행을 올바르게 가져오기 위해 iloc 사용
            content = safe_get(self.data.iloc[i], 'top-1(ENG)')
            if content:
                content_list.append(content)

        content_list += [''] * (5 - len(content_list))
        content_list_str = ', '.join(content_list)

        # Stage 3: Content Recommendation
        stage_3_input = f"{stage_1_input}\n\n{stage_1_output}\n\n{stage_2_input}\n\n{stage_2_output}\n\n{STAGE_3_TEMPLATE.format(content_list=content_list_str)}"

        if self.add_token:
            stage_3_input += f" {self.new_token} "

        encodings = self.tokenizer(stage_3_input, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        # return {
        #     **{key: val.squeeze(0) for key, val in encodings.items()},
        #     'recommendation_embeddings': self.recommendation_embeddings[idx],
        #     'labels': torch.tensor(item.get('label', 0), dtype=torch.long)
        # }
        return {
            **{key: val.squeeze(0) for key, val in encodings.items()},
            'recommendation_embeddings': self.recommendation_embeddings[idx],  # 이는 40x768 형태여야 합니다
        }
    
class RecommenderTrainer:
    
    def __init__(self, model, tokenizer, data, device, qlora_params=None):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data
        self.device = device
        self.dataset = None
        self.dataloader = None
        self.qlora_wrapper = QLoRAWrapper(**(qlora_params or {}))
    
    def prepare_model_for_training(self):
        self.model = self.qlora_wrapper.prepare_model(self.model)

    def prepare_dataset(self, csv_file_path, recommendation_embeddings):
        # self.dataset = RecommenderDataset(self.data, self.tokenizer, self.tokenizer.add_token, self.tokenizer.new_token)
        # self.dataloader = DataLoader(self.dataset, batch_size=12, shuffle=True)
        self.dataset = RecommenderDataset(self.data, self.tokenizer, recommendation_embeddings)
        self.dataloader = DataLoader(self.dataset, batch_size=12, shuffle=True)
        # NOTE: Need to change the batch_size(It was actually set to in a large dataset 12), maybe add some more augmentations
    
    # NOTE: The num_epochs were set to 4 in the original platform
    '''
        1. Need to change the parameter into a config yaml file
    '''
    def train(self, num_epochs=4):
        self.prepare_model_for_training()
        
        optimizer = bnb.optim.AdamW8bit(self.model.parameters(), lr=0.0002)
        
        total_steps = len(self.dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
        
        criterion = torch.nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in self.dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Ouput Token
                outputs = self.model(batch['input_ids'], batch['attention_mask'], batch['recommendation_embeddings'])
                
                # 손실 Cross_entropy
                # outputs: [batch_size, sequence_length, vocab_size]
                # batch['input_ids']: [batch_size, sequence_length]
                loss = criterion(outputs.view(-1, outputs.size(-1)), batch['input_ids'].view(-1))
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
                
    def save_model(self, path):
        # Save the model with merged weights
        self.model.save_pretrained(f"{path}/model")
        self.tokenizer.save_pretrained(f"{path}/tokenizer")
        
        # NOTE: After training, you might want to merge the LoRA weights with the base model
        # This step is crucial for using the model with VLLM later
        # Example (pseudo-code):
        # merged_model = merge_lora_weights(self.model)
        # merged_model.save_pretrained(f"{path}/merged_model")
        
    def inference(self, patient_data, content_list):
        # Stage 1: Patient Analysis
        stage_1_input = STAGE_1_TEMPLATE.format(**patient_data)
        stage_1_inputs = self.tokenizer(stage_1_input, return_tensors="pt", truncation=True, padding=True).to(self.device)
        stage_1_output = self.model.generate(**stage_1_inputs)
        stage_1_output = self.tokenizer.decode(stage_1_output[0], skip_special_tokens=True)

        # Stage 2: Management Plan
        stage_2_input = f"{stage_1_input}\n\n{stage_1_output}\n\n{STAGE_2_TEMPLATE}"
        stage_2_inputs = self.tokenizer(stage_2_input, return_tensors="pt", truncation=True, padding=True).to(self.device)
        stage_2_output = self.model.generate(**stage_2_inputs)
        stage_2_output = self.tokenizer.decode(stage_2_output[0], skip_special_tokens=True)

        # Stage 3: Content Recommendation
        stage_3_input = f"{stage_1_input}\n\n{stage_1_output}\n\n{stage_2_input}\n\n{stage_2_output}\n\n{STAGE_3_TEMPLATE.format(content_list=content_list)}"
        stage_3_inputs = self.tokenizer(stage_3_input, return_tensors="pt", truncation=True, padding=True).to(self.device)
        stage_3_output = self.model.generate(**stage_3_inputs)
        
        return self.tokenizer.decode(stage_3_output[0], skip_special_tokens=True)

# NOTE: After training, when loading the model for inference with VLLM:
# 1. Load the merged model (if you merged the weights after training)
# 2. Use VLLM's LLM class to load the model for efficient inference
# Example (pseudo-code):
# from vllm import LLM
# vllm_model = LLM(model_path=f"{path}/merged_model")