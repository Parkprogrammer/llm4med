import torch
from vllm import LLM
from parse import RecommenderTokenizer
    
class RecommenderModel:
    
    def __init__(self, model_id):
        self.model_id = model_id
        self.vllm_model = None
        self.tokenizer = None
        
    def load_model(self):
        self.vllm_model = LLM(model=self.model_id, trust_remote_code=True)
        self.tokenizer = RecommenderTokenizer(self.model_id)
        self.tokenizer.load_tokenizer()
        
    def generate_recommendation(self, user_preferences):
        llm_chain = self.tokenizer.create_llm_chain(self.vllm_model)
        return llm_chain.run(preferences=user_preferences)
    
        # Not attention mask, a single head for softmax and linear layer
        # TODO: Add a softmax-MLP output head