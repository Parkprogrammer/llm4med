import safetensors.torch
import torch

# 분할된 safetensors 파일 경로
safetensor_files = [
    "./saved_model/model-00001-of-00005.safetensors",
    "./saved_model/model-00002-of-00005.safetensors",
    "./saved_model/model-00003-of-00005.safetensors",
    "./saved_model/model-00004-of-00005.safetensors",
]

# 병합된 파일을 저장할 경로
output_file = "./saved_model/model.safetensors"

# 모든 텐서를 병합
tensors = {}
for file in safetensor_files:
    loaded_tensors = safetensors.torch.load_file(file)
    tensors.update(loaded_tensors)

# 병합된 결과를 model.safetensors로 저장
safetensors.torch.save_file(tensors, output_file)
print(f"Merged model saved at {output_file}")
