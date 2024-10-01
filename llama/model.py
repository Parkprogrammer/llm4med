import torch
import torch.nn as nn
import torch.nn.functional as F
from .parse import RecommenderTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

''' Changed Method for item embeddings 

class RecommendationTokenGenerator(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=768):
        super(RecommendationTokenGenerator, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.projector = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.projector(x)
        return x

'''
    
class OutputMLP(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=1024, output_dim=768, sequence_length=40):
        super(OutputMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim * sequence_length)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x.view(-1, self.sequence_length, self.output_dim)
    
class OutputMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=768, sequence_length=40):
        super(OutputMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim * sequence_length)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x.view(-1, self.sequence_length, self.output_dim)

class RecommenderModel(nn.Module):
    def __init__(self, model_id, tokenizer):
        super(RecommenderModel, self).__init__()
        self.model_id = model_id
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            load_in_8bit=True,         
            torch_dtype=torch.float16  
        )
        self.tokenizer = tokenizer
        self.output_head = OutputMLP(input_dim=768, output_dim=768, sequence_length=40)
        
    def forward(self, input_ids, attention_mask, recommendation_embeddings):
        base_outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = base_outputs.hidden_states[-1]
        
        emb_token_id = self.tokenizer.convert_tokens_to_ids('[EMB]')
        for i in range(input_ids.size(0)):
            emb_positions = torch.where(input_ids[i] == emb_token_id)[0]
            
            for j, pos in enumerate(emb_positions):
                if j < 5:  # 5개 이상의 아이템이 있을 경우를 대비
                    last_hidden_state[i, pos:pos+40, :] = recommendation_embeddings[i, j]

        pooled_output = last_hidden_state.mean(dim=1)  # Simple mean pooling
        
        return self.output_head(pooled_output)

    def generate_recommendation(self, user_preferences, recommendation_embeddings, max_length=512):
        inputs = self.tokenizer(user_preferences, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(self.base_model.device)
        attention_mask = inputs["attention_mask"].to(self.base_model.device)
        
        with torch.no_grad():
            base_outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                output_hidden_states=True
            )
        
        last_hidden_state = self.base_model.transformer.h[-1](base_outputs[:, -1:, :]).last_hidden_state
        pooled_output = last_hidden_state.mean(dim=1)
        
        mlp_output = self.output_head(pooled_output)
        
        return mlp_output, self.tokenizer.decode(base_outputs[0], skip_special_tokens=True)

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **kwargs):
        return self.base_model.prepare_inputs_for_generation(input_ids, past, attention_mask, **kwargs)

    def get_tokenizer(self):
        return self.tokenizer
    
        # Not attention mask, a single head for softmax and linear layer
        # TODO: Add a softmax-MLP output head