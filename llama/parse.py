import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from .template import STAGE_1_TEMPLATE, STAGE_2_TEMPLATE, STAGE_3_TEMPLATE
import torch

os.environ['HF_HOME'] = '/data/transformer_cache'

class RecommenderTokenizer:
    
    def __init__(self, model_id, add_token=False, new_token="[RECOMMEND]"):
        self.model_id = model_id
        self.add_token = add_token
        self.new_token = new_token
        self.tokenizer = None
        self.model = None
        
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        if self.add_token:
            self.tokenizer.add_tokens([self.new_token])
            # self.model.resize_token_embeddings(len(self.tokenizer))
        # NOTE: This is where the actual adding token took place.
    
    def create_llm_chain(self):
        # Create a HuggingFacePipeline
        pipeline = HuggingFacePipeline(pipeline=self.model)
        
        stage_1_prompt = PromptTemplate(template=STAGE_1_TEMPLATE, input_variables=["disease_classification", "departments", "systems", "symptoms_complications", "gender", "age", "bmi"])
        stage_2_prompt = PromptTemplate(template=STAGE_2_TEMPLATE, input_variables=[])
        stage_3_prompt = PromptTemplate(template=STAGE_3_TEMPLATE, input_variables=["top-1(ENG)"])
        
        stage_1_chain = LLMChain(llm=pipeline, prompt=stage_1_prompt)
        stage_2_chain = LLMChain(llm=pipeline, prompt=stage_2_prompt)
        stage_3_chain = LLMChain(llm=pipeline, prompt=stage_3_prompt)
        
        return stage_1_chain, stage_2_chain, stage_3_chain
    
    def tokenize(self, text, max_length=512):
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
    def process_chain_of_thought(self, patient_data, content_list):
        stage_1_chain, stage_2_chain, stage_3_chain = self.create_llm_chain()
        
        # Stage 1: Patient Analysis
        stage_1_output = stage_1_chain.run(patient_data)
        
        # Stage 2: Management Plan
        stage_2_output = stage_2_chain.run(stage_1_output)
        
        # Stage 3: Content Recommendation
        stage_3_input = f"{stage_1_output}\n\n{stage_2_output}\n\nAvailable content:\n{content_list}"
        final_recommendations = stage_3_chain.run(content_list=stage_3_input)
        
        return final_recommendations

    def generate(self, prompt, max_length=512):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)