# LLM4Med: AI-Based Medical Educational Resource Recommendation System




## Project Overview

LLM4Med is an AI-powered web application that analyzes patient data and recommends customized educational resources. This system utilizes a medical-specialized Large Language Model (LLM) to perform in-depth analysis of patient information and recommends relevant educational materials based on the analysis results.<br/>
__*Project was intergrated into iKooB Healthcare platform with upgraded versions__

### Key Features

1. **Multi-stage Patient Data Analysis**:
   - **Stage 1**: Analysis of patient health status and key management areas
   - **Stage 2**: Development of disease management plans, complication prevention strategies, and lifestyle improvement recommendations
   - **Stage 3**: Provision of personalized educational topics and actionable advice

2. **Educational Resource Recommendation**: TF-IDF-based keyword matching for relevant educational material recommendations

3. **Multilingual Support**: Automatic translation of English analysis results to Korean

## System Requirements

### Hardware Requirements
- **GPU**: 12GB+ VRAM (recommended for running Llama-3 models)
- **RAM**: 16GB+
- **Storage**: 15GB+ (for model and data storage)

### Software Requirements
- **Python**: 3.8+
- **CUDA**: 11.7+ (for GPU usage)

## Installation

### 1. Environment Setup

```bash
# Create a virtual environment
conda create -n llm4med python=3.8

# Activate the virtual environment
conda activate llm4med

# Install required packages
pip install -r requirements.txt
```

### 2. Directory and File Setup

```bash
# Create directories for images and educational resources
mkdir -p content/images

# Create ngrok configuration file (optional: for external access)
echo '{"auth_token": "your_ngrok_auth_token", "domain": "your_ngrok_domain"}' > ngrok_config.json
```

### 3. Papago API Key Setup (for Korean translation)

Set as environment variables:
```bash
export PAPAGO_CLIENT_ID="your_client_id"
export PAPAGO_CLIENT_SECRET="your_client_secret"
```

Alternatively, you can directly set them in `utils.py`.

## File Structure and Required Modifications

### Key Files

| Filename | Description |
|----------|-------------|
| `main.py` | Main Flask application and API endpoint definitions |
| `model.py` | LLM model and processor class definitions |
| `Recsys.py` | TF-IDF-based recommendation algorithm for educational resources |
| `template.py` | LLM prompt templates and HTML template definitions |
| `utils.py` | Utility functions (translation, ngrok setup, etc.) |

### Required Modifications

#### 1. Path Settings in `main.py`

The following variables need to be directly set:

```python
# Directory containing content images
IMAGE_DIRECTORY = "/path/to/your/image/directory"

# Path to the educational resource metadata CSV file
RECOMMENDATION_PATH = "/path/to/your/recommendation.csv"
```

#### 2. Papago API Key Setup in `utils.py`

```python
def translate_to_korean(text, client_id="YOUR_PAPAGO_CLIENT_ID", client_secret="YOUR_PAPAGO_CLIENT_SECRET"):
```

#### 3. Patient Data CSV Format

The CSV file uploaded to the system should include the following fields:

- **Required fields**:
  - `Disease Classification - Primary`, `Disease Classification - Secondary`, `Disease Classification - Tertiary` (disease classifications)
  - `Department - Main`, `Department - Sub` (related departments)
  - `System`, `System.1`, `System.2`, `System.3`, `System.4` (affected systems)
  - `Disease Name`, `Disease Name.1`, `Disease Name.2`, `Disease Name.3`, `Disease Name.4` (disease names/symptoms)
  - `Gender`, `Age`, `BMI` (patient characteristics)
  - `top-1(ENG)` (recommended educational topic)

#### 4. Educational Resource Metadata CSV Format

The CSV file specified in `RECOMMENDATION_PATH` should follow this format:

- **Required fields**:
  - `No` (unique identifier)
  - `Title` (educational resource title)
  - `File_name` (image filename)
  - `Keywords` (keyword list - string or list format)

## Execution

### Basic Execution

```bash
python main.py
```

By default, the web application runs at `http://localhost:5000`.

### External Access Setup via ngrok (Optional)

To create a URL accessible from outside, configure the `ngrok_config.json` file and run:

```json
{
  "auth_token": "your_ngrok_auth_token",
  "domain": "your_custom_domain" (optional)
}
```

## Usage Guide

1. Access the web interface (`http://localhost:5000`)
2. Upload a CSV file containing patient data
3. Select a patient to analyze from the patient list
4. The multi-stage analysis will automatically proceed:
   - Stage 1: Patient information analysis
   - Stage 2: Management plan development
   - Stage 3: Detailed recommendation generation
5. View the analysis results along with recommended educational resources

## Dependencies

The `requirements.txt` file should include the following packages:

```
flask==2.0.2
vllm==0.2.0
pandas==1.5.3
numpy==1.24.3
pyngrok==5.2.1
langchain==0.0.230
scikit-learn==1.2.2
torch==2.1.0
```

## Troubleshooting

### Common Errors

1. **Model Loading Error**
   - Issue: `LLM` object initialization failure
   - Solution: Check GPU VRAM, verify CUDA compatibility

2. **File Upload Error**
   - Issue: CSV file format error
   - Solution: Ensure CSV field names match the expected format

3. **Image Loading Failure**
   - Issue: Recommended images not displaying
   - Solution: Check `IMAGE_DIRECTORY` path, verify file extensions (.png)

4. **Translation API Error**
   - Issue: Korean translation failure
   - Solution: Verify Papago API keys, check network connectivity

## Model Information

This project uses the TsinghuaC3I/Llama-3-8B-UltraMedical model, which is an 8B parameter Llama-3 model fine-tuned on medical data.

- **Model Source**: [Hugging Face - TsinghuaC3I/Llama-3-8B-UltraMedical](https://huggingface.co/TsinghuaC3I/Llama-3-8B-UltraMedical)
- **License**: Check the license before using the model

## Code Structure and Workflow

### Data Flow
1. Patient CSV data upload (`/upload` endpoint)
2. Patient list provision and pagination (`/patients_list` endpoint)
3. Three-stage analysis of selected patient (`/process_stage` endpoint)
4. Educational resource recommendation and display based on analysis results

### Class and Function Relationships
- `LangchainCOTProcessor`: LLM interaction management
- `prepare_input_data()`: Patient data preprocessing
- `extract_keywords_from_output()`: Keyword extraction
- `find_best_matches_tfidf()`: Educational resource matching
- `translate_to_korean()`: Translation processing

## Extension and Improvement Directions

1. **LLM Model Replacement**:
   - Change the `model_name` parameter of the `LangchainCOTProcessor` class in `model.py`
   - Adjust tokenizer and generation parameters as needed

2. **Prompt Template Modification**:
   - Modify `STAGE_1_TEMPLATE`, `STAGE_2_TEMPLATE`, `STAGE_3_TEMPLATE` in `template.py`

3. **UI Improvements**:
   - Modify the `HTML_TEMPLATE` in `template.py`
   - Add CSS and JavaScript functionality

4. **Recommendation Algorithm Enhancement**:
   - Consider implementing embedding-based recommendation algorithms instead of TF-IDF in `Recsys.py`

## License

Please check Llama license before use.

## Contact

For additional information, please email jaheon555@g.skku.edu