from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class QLoRAWrapper:
    # Default settings from https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B
    def __init__(self, r=128, lora_alpha=256, lora_dropout=0.05):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        
    def prepare_model(self, model):
        
        # NOTE: This is where the wrapping with LoRA begins
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r = self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",    # This is part which is unknown to me.
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        return model