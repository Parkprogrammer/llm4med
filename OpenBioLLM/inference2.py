import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['TRANSFORMERS_CACHE'] = '/data/huggingface_cache'

def load_model_and_tokenizer():
    model_name = "./saved_model/models--aaditya--Llama3-OpenBioLLM-70B/snapshots/5f79deaf38bc5f662943d304d59cb30357e8e5bd"
    
    # 4-bit quantization settings
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # Memory and device map settings
    max_memory = {0: "22GB", "cpu": "58GB"}  # Adjust based on your hardware capabilities
    device_map = "auto"  # Use "auto" to let Hugging Face handle optimal device placement
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch.float16,
        offload_folder="/data/model_offload",  # Ensure this folder exists
        load_in_8bit_fp32_cpu_offload=True  # Allow CPU offloading in 32-bit precision
    )
    
    return model, tokenizer
    
    return model, tokenizer

def create_inference_pipeline(model, tokenizer):
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0  # Ensure the pipeline is using the GPU
    )

def generate_recommendation(pipe, user_preferences, max_length=200):
    prompt = f"You are an expert recommender system. Given the following user preferences, recommend a suitable item.\nUser preferences: {user_preferences}\nRecommendation:"
    
    print("Generating recommendation...")
    output = pipe(prompt, max_length=max_length, num_return_sequences=1, do_sample=True, temperature=0.7, truncation=True)
    
    generated_text = output[0]['generated_text']
    recommendation = generated_text.split("Recommendation:")[1].strip() if "Recommendation:" in generated_text else generated_text
    
    return recommendation

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    
    inference_pipeline = create_inference_pipeline(model, tokenizer)
    
    # Test user preference
    user_preferences = "I am looking for a treatment for chronic lower back pain that doesn't involve surgery."
    
    recommendation = generate_recommendation(inference_pipeline, user_preferences)
    
    print(f"\nUser Preferences: {user_preferences}")
    print(f"\nRecommendation: {recommendation}")
