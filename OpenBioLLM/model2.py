import torch
from transformers import AutoModelForCausalLM
from langchain import HuggingFacePipeline

class RecommenderModel:
    
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = None
        self.llm = None
        
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype = torch.bfloat16,   # Used bfloat16 for large number of parameters                            
            device_map = "auto",            # Accelaration Huggin Face labotory
            # load_in_8bit = True           # Saving for efficiency but NOT NOW
        )
        
    def create_langchain_pipeline(self, tokenizer):
        self.llm = HuggingFacePipeline.from_model_id(
            model_id = self.model_id,
            task = "text-generation",       # Recommending task avialable?
            model = self.model,
            tokenizer = tokenizer,          # But bfloat16 normalization?
            model_kwargs = {"torch_dtype" : torch.bfloat16, "device_map" : "auto"},
        )        
        
    def forward(self, input_ids, attention_mask=None, labels=None): # Not attention mask, a single head for sofmax and linear layer
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # TODO: Add a softmax-MLP output head
