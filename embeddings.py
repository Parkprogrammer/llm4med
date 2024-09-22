import torch
from KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import BertTokenizer, AutoTokenizer
from kobert_tokenizer import KoBERTTokenizer
import torch.nn.functional as F

def get_embeddings(input_text):
    """
    Extract embeddings from input phrases using KoBERT.
    
    Args:
    - input_text (str): Input text with phrases separated by commas.
    
    Returns:
    - torch.Tensor: Embeddings of the input phrases.
    """
    model, vocab = get_pytorch_kobert_model()
    tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
    phrases = [phrase.strip() for phrase in input_text.split(',')]
    embeddings = []
    
    for phrase in phrases:
        inputs = tokenizer(phrase, return_tensors='pt', padding='max_length', max_length=20, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = torch.zeros_like(input_ids)
        
        with torch.no_grad():
            _, pooled_output = model(input_ids, attention_mask, token_type_ids)
        
        embeddings.append(pooled_output.squeeze(0))
    
    return torch.stack(embeddings)

def calculate_cosine_similarity(embeddings):
    """
    Calculate cosine similarity matrix for the given embeddings.
    
    Args:
    - embeddings (torch.Tensor): The embeddings to calculate similarity for.
    
    Returns:
    - torch.Tensor: Cosine similarity matrix.
    """
    num_phrases = embeddings.size(0)
    similarity_matrix = torch.zeros((num_phrases, num_phrases))
    
    for i in range(num_phrases):
        for j in range(num_phrases):
            similarity = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
            similarity_matrix[i, j] = similarity.item()
    
    return similarity_matrix

def main():
    """Made this for both making embeddings and calculating similarity"""
    input_text = "회사 데이터"
    embeddings = get_embeddings(input_text)
    
    '''
    similarity_matrix = calculate_cosine_similarity(embeddings)
    
    similarity_matrix_filepath = "./data/cosine_similarity_matrix.txt"
    
    with open(similarity_matrix_filepath, 'w') as f:
        for row in similarity_matrix:
            f.write("\t".join([f"{value:.4f}" for value in row.tolist()]) + "\n")
    
    print(f"Cosine similarity matrix has been saved to {similarity_matrix_filepath}")
    '''

if __name__ == "__main__":
    main()