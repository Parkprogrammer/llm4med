import torch
from KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import BertTokenizer, AutoTokenizer
from kobert_tokenizer import KoBERTTokenizer
import torch.nn.functional as F
import pandas as pd


def get_embeddings(input_text, model, tokenizer):
    """
    Extract embeddings from input phrases using KoBERT and pad/truncate to length 40.

    Args:
    - input_text (str): Input text with phrases separated by commas.

    Returns:
    - torch.Tensor: 40 x 768 embedding matrix of the input phrases (including padding if needed).
    """
    # input_text가 NaN이거나 float인 경우 빈 문자열로 변환
    if not isinstance(input_text, str):
        input_text = ""

    phrases = [phrase.strip() for phrase in input_text.split(',') if phrase.strip()]
    embeddings = []
    
    for phrase in phrases:
        inputs = tokenizer(phrase, return_tensors='pt', padding='max_length', max_length=40, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = torch.zeros_like(input_ids)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, token_type_ids)
            pooled_output = outputs[0]  # outputs[0]은 마지막 hidden state로, (batch_size, sequence_length, hidden_size) 형태
            
            # phrase에 대한 임베딩의 평균을 구함
            phrase_embedding = pooled_output.mean(dim=1).squeeze(0)
        
        embeddings.append(phrase_embedding)
    
    # 필요한 임베딩이 40개보다 적을 경우 패딩을 추가하고, 40개를 넘으면 잘라냄
    if len(embeddings) < 40:
        padding = [torch.zeros(768) for _ in range(40 - len(embeddings))]
        embeddings.extend(padding)
    else:
        embeddings = embeddings[:40]
    
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

def test():
    """Made this for both making embeddings and calculating similarity"""
    # Load the pre-trained KoBERT model and tokenizer
    model, _ = get_pytorch_kobert_model()
    tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
    
    input_text = "회사 데이터"
    embeddings = get_embeddings(input_text, model, tokenizer)
    print(embeddings.shape)
    
    '''
    similarity_matrix = calculate_cosine_similarity(embeddings)
    
    similarity_matrix_filepath = "./data/cosine_similarity_matrix.txt"
    
    with open(similarity_matrix_filepath, 'w') as f:
        for row in similarity_matrix:
            f.write("\t".join([f"{value:.4f}" for value in row.tolist()]) + "\n")
    
    print(f"Cosine similarity matrix has been saved to {similarity_matrix_filepath}")
    '''

def main():
    
    # Load the pre-trained KoBERT model and tokenizer
    model, _ = get_pytorch_kobert_model()
    tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')

    # Read the input CSV file
    df = pd.read_csv('./data/embedding.csv')

    # Initialize a list to store all embeddings for the final .pt file
    all_embeddings = []

    # Create embeddings for each row in the dataframe
    for index, row in df.iterrows():
        top_k_text = row['Top-k']
        embedding_tensor = get_embeddings(top_k_text, model, tokenizer)
        
        # Add the tensor to the list for the .pt file
        all_embeddings.append(embedding_tensor.unsqueeze(0))

    # Concatenate all tensors to create the final N x 40 x 768 matrix
    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    
    # Save the result
    torch.save(all_embeddings_tensor, './data/all_embeddings.pt')
    print("Embedding process completed. File saved as 'all_embeddings.pt'.")






if __name__ == "__main__":
    
    # input_text = "회사 데이터"
    # embeddings = get_embeddings(input_text)
    
    # main()
    # test()
    
    all_embeddings_tensor = torch.load('./data/all_embeddings.pt')
    
    # Check the size of the tensor
    print("The size of the loaded tensor is:", all_embeddings_tensor.size())
    # print(all_embeddings_tensor[0][11])