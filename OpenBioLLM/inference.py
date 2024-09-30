import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_id):
    # 저장된 모델 경로
    saved_model_path = "./saved_model"

    # Hugging Face 토큰
    hf_token = "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"  # 본인의 Hugging Face 토큰으로 변경하세요

    if os.path.exists(saved_model_path):
        print(f"Loading model and tokenizer from local path {saved_model_path}...")
        
        # Check if the tokenizer files exist, if not download again
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        missing_files = [f for f in tokenizer_files if not os.path.exists(os.path.join(saved_model_path, f))]
        
        if missing_files:
            print(f"Missing tokenizer files: {missing_files}. Re-downloading tokenizer.")
            tokenizer = AutoTokenizer.from_pretrained(model_id, resume_download=True, token=hf_token)
            tokenizer.save_pretrained(saved_model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(saved_model_path)

        # Load the model using the safetensors format
        model = AutoModelForCausalLM.from_pretrained(
            saved_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="./offload_weights",
            use_safetensors=True,   # Add this option to handle safetensors format
            trust_remote_code=True,  # Add this to allow loading with multiple shard support
            token=hf_token           # Use token for authentication
        )
    else:
        print(f"Downloading model and tokenizer from {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        
        # Specify an offload folder for disk storage
        offload_folder = "./offload_weights"
        os.makedirs(offload_folder, exist_ok=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder=offload_folder,
            use_safetensors=True,
            trust_remote_code=True,
            token=hf_token  # Use token for authentication
        )

        # 모델과 토크나이저를 safetensors 형식으로 저장
        print(f"Saving model and tokenizer to {saved_model_path} with safe serialization...")
        os.makedirs(saved_model_path, exist_ok=True)
        
        model.save_pretrained(saved_model_path, safe_serialization=True)
        tokenizer.save_pretrained(saved_model_path)

    print("Model and tokenizer loaded successfully.")
    return model, tokenizer





def create_inference_pipeline(model, tokenizer):
    # Removed the 'device' argument as it conflicts with accelerate
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

def generate_recommendation(pipe, user_preferences, max_length=200):
    prompt = f"You are an expert recommender system. Given the following user preferences, recommend a suitable item.\nUser preferences: {user_preferences}\nRecommendation:"
    
    print("Generating recommendation...")
    output = pipe(prompt, max_length=max_length, num_return_sequences=1, do_sample=True, temperature=0.7, truncation=True)
    
    generated_text = output[0]['generated_text']
    recommendation = generated_text.split("Recommendation:")[1].strip() if "Recommendation:" in generated_text else generated_text
    
    return recommendation

if __name__ == "__main__":
    model_id = "aaditya/Llama3-OpenBioLLM-70B"
    
    model, tokenizer = load_model_and_tokenizer(model_id)
    
    inference_pipeline = create_inference_pipeline(model, tokenizer)
    
    # 테스트용 사용자 선호도
    user_preferences = "I am looking for a treatment for chronic lower back pain that doesn't involve surgery."
    
    recommendation = generate_recommendation(inference_pipeline, user_preferences)
    
    print(f"\nUser Preferences: {user_preferences}")
    print(f"\nRecommendation: {recommendation}")
