from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import VLLMOpenAI
from vllm import LLM, SamplingParams
import os

os.environ['TRANSFORMERS_CACHE'] = '/data/transformer_cache'

class RecommenderTokenizer:
    
    def __init__(self, model_id, add_token=False, new_token="[RECOMMEND]"):
        self.model_id = model_id
        self.add_token = add_token
        self.new_token = new_token
        self.tokenizer = None
        
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.add_token:
            self.tokenizer.add_tokens([self.new_token]) 
        # NOTE: This is where the actual adding token took place.
    
    def create_llm_chain(self, vllm_model):
        # TODO: Make the actual template!! -> Need some prompt-tuning
        '''
            1. Added a long sentence for explaining the role?
            2. (ing...)
        '''
        
        if self.add_token:
            template = """You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience. 
            Given the following user preferences, recommend a suitable item.
            User preferences: {preferences}
            [RECOMMEND]
            Recommendation:"""
        else:
            template = """You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience.
            Given the following user preferences, recommend a suitable item.
            User preferences: {preferences}
            Recommendation:"""
        
        prompt = PromptTemplate(template=template, input_variables=["preferences"])
        
        # Wrap VLLM model in LangChain's VLLMOpenAI
        llm = VLLMOpenAI(vllm_model=vllm_model)
        
        return LLMChain(llm=llm, prompt=prompt)
    
    # NOTE: Need some verifications about this one with checking
    def tokenize(self, text, max_length=512):
        
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )