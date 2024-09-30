import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# TRANSFORMERS_CACHE 환경 변수 설정
os.environ['HF_HOME'] = '/data/huggingface_cache'

def load_model_and_tokenizer():
    model_name = "TsinghuaC3I/Llama-3-8B-UltraMedical"
    
    llm = LLM(model=model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return llm, tokenizer

def generate_recommendation(llm, tokenizer, user_preferences):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024, stop=["<|eot_id|>"])
    
    messages = [
        {"role": "user", "content": f"You are an expert medical recommender system. Given the following user preferences, recommend a suitable treatment or advice.\nUser preferences: {user_preferences}\nRecommendation:"}
    ]
    
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    
    recommendation = outputs[0].outputs[0].text.strip()
    return recommendation

if __name__ == "__main__":
    llm, tokenizer = load_model_and_tokenizer()
    
    # 테스트용 사용자 선호도
    user_preferences = """
    Please analyze the following patient information:
    - Disease classification: Endocrine, nutritional and metabolic diseases (diabetes: type 1, type 2, other specific diabetes)
    - Related departments: Internal Medicine, Family Medicine
    - Affected systems: endocrine system, cardiovascular system, renal/urinary system, ophthalmology/otolaryngology
    - Potential symptoms/complications: hyperglycemia, retinopathy, peripheral vascular disease, neuropathy
    - General characteristics: Gender (Both), Age (Common), BMI (Common)

    Based on this information, please summarize the patient's main health concerns and areas requiring care.
    """
    
    recommendation = generate_recommendation(llm, tokenizer, user_preferences)
    
    print(f"\nUser Preferences: {user_preferences}")
    print(f"\nRecommendation: {recommendation}")