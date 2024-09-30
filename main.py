import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from llama.model import RecommenderModel
from llama.parse import RecommenderTokenizer
from llama.trainer import RecommenderTrainer, RecommenderDataset
import os

os.environ['HF_HOME'] = '/data/transformer_cache'

def setup_environment(model_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recommender_tokenizer = RecommenderTokenizer(model_id, add_token=True, new_token="[RECOMMEND]")
    recommender_tokenizer.load_tokenizer()
    model = RecommenderModel(model_id, recommender_tokenizer.tokenizer)
    return device, recommender_tokenizer, model

def load_data(file_path):
    return pd.read_csv(file_path)

def load_embeddings(file_path):
    try:
        embeddings = torch.load(file_path)
        if embeddings.shape != (1365, 40, 768):
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}. Expected (1365, 40, 768)")
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

def prepare_dataset(data, tokenizer, recommendation_embeddings, batch_size=12):
    dataset = RecommenderDataset(data, tokenizer, recommendation_embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, dataloader

def setup_trainer(model, tokenizer, data, device, dataset, dataloader):
    trainer = RecommenderTrainer(model, tokenizer, data, device)
    trainer.dataset = dataset
    trainer.dataloader = dataloader
    return trainer

def train_model(trainer, num_epochs, qlora_params):
    # trainer.prepare_model_for_training()
    trainer.train(num_epochs=num_epochs)

def save_model(trainer, path):
    trainer.save_model(path)
    print("Model saved to:", path)

def run_inference(trainer, patient_data, content_list):
    recommendation = trainer.inference(patient_data, content_list)
    print("Generated Recommendation:", recommendation)

def main():
    # Configuration
    model_id = "TsinghuaC3I/Llama-3-8B-UltraMedical"
    data_path = "./data/final_data.csv"
    model_save_path = "./train"
    num_epochs = 4
    qlora_params = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }

    # Setup
    device, recommender_tokenizer, model = setup_environment(model_id)
    
    # Data preparation
    data = load_data(data_path)
    recommendation_embeddings = load_embeddings("./data/all_embeddings.pt")
    dataset, dataloader = prepare_dataset(data, recommender_tokenizer.tokenizer, recommendation_embeddings)

    # Trainer setup and training
    trainer = setup_trainer(model, recommender_tokenizer.tokenizer, data, device, dataset, dataloader)
    train_model(trainer, num_epochs, qlora_params)
    
    # Save model
    save_model(trainer, model_save_path)

    # Inference example
    # patient_data = {
    #     "disease_classification": "Type 2 Diabetes",
    #     "departments": "Endocrinology, Internal Medicine",
    #     "systems": "Endocrine, Cardiovascular",
    #     "symptoms_complications": "Hyperglycemia, Neuropathy",
    #     "gender": "Male",
    #     "age": "45",
    #     "bmi": "28"
    # }
    # content_list = "Diet plan, Exercise routine, Medication guide, Glucose monitoring tips"
    # run_inference(trainer, patient_data, content_list)
    
    # user_preferences = "Some user preferences..."
    # recommendation_embeddings = torch.randn(1, 40, 768)
    # mlp_output, generated_text = model.generate_recommendation(user_preferences, recommendation_embeddings)
    
    # print("MLP Output:", mlp_output.shape)
    # print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()