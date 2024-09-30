from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the model ID from Hugging Face Hub
model_id = "aaditya/Llama3-OpenBioLLM-70B"

# The path to the directory where the model should be saved
saved_model_path = "./saved_model"

print("Downloading the model and tokenizer...")

# Attempt to re-download the model using force_download=True
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=saved_model_path,
    resume_download=True,   # Resume any incomplete downloads
    force_download=True     # Force re-download to ensure the missing file is retrieved
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=saved_model_path,
    resume_download=True,   
    force_download=True
)

print("Download complete. The model and tokenizer should now be fully available in the saved_model directory.")
