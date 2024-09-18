import torch
from OpenBioLLM.model import RecommenderModel
from OpenBioLLM.parse import RecommenderTokenizer
from OpenBioLLM.trainer import RecommenderTrainer
from OpenBioLLM.trainer import RecommenderDataset
from OpenBioLLM.qlora import QLoRAWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    
    model_id = "aaditya/OpenBioLLM-Llama3-70B"
    
    recommender_model = RecommenderModel(model_id)
    recommender_model.load_model()
    
    tokenizer = RecommenderTokenizer(model_id,True,"[RECOMMEND]")
    
    # TODO: Import data using MobaXTerm
    data = ...
    
    # TODO: Add the KoBERT and use it for adjusting template(Do it here?)
    trainer = RecommenderTrainer(recommender_model, tokenizer, data, device)
    trainer.prepare_dataset()
    trainer.train()
    trainer.save_model("./train")