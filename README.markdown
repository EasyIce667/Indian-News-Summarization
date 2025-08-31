# Indian News Summarization Project

This project demonstrates text summarization of Hindi news articles using a fine-tuned **IndicBARTSS** model from `ai4bharat`. The pipeline includes data preprocessing, model fine-tuning, evaluation, and an interactive Gradio interface for generating summaries.

## Overview
This project uses the **IndicBARTSS** model, a sequence-to-sequence transformer designed for Indian languages, to summarize Hindi news articles. The workflow includes:
- Loading and preprocessing a Hindi news dataset.
- Fine-tuning the IndicBARTSS model.
- Testing the model on a test set.
- Deploying an interactive Gradio interface for real-time summarization.

## Dataset
The dataset (`hindi_news_dataset.csv`) contains Hindi news articles with the following columns:
- `Headline`: The article’s headline (used as the reference summary).
- `Content`: The full article text.
- `News Categories`: Categories like national, politics, etc.
- `Date`: Publication date.

The dataset is preprocessed to clean text, remove empty rows, and tokenize data. A subset of 5,000 samples is used (80% train, 20% test) to manage computational resources.

**Note**: The dataset is not included in this repository. You must obtain `hindi_news_dataset.csv` and place it in the project directory or Google Drive (`/content/drive/My Drive/`).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/indian-news-summarization.git
   ```

2. Install dependencies:
   ```bash
   pip install transformers==4.40
   pip install torch==2.0.0
   pip install indic-nlp-library==0.92
   pip install sentencepiece==0.2.0
   pip install gradio==4.38.0
   pip install datasets==2.21.0
   pip install pandas==2.2.2
   pip install wandb==0.19.11
   ```

3. (Optional) Mount Google Drive in Google Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. Set up Weights & Biases (WandB) for training logging:
   - Create a WandB account and obtain an API key from [https://wandb.ai](https://wandb.ai).
   - Set the API key as an environment variable or enter it when prompted during training:
     ```bash
     export WANDB_API_KEY=your_wandb_api_key
     ```

## Usage
1. **Prepare the Dataset**:
   - Ensure `hindi_news_dataset.csv` is available in the project directory or at `/content/drive/My Drive/`.
   - The notebook includes error handling for loading issues (e.g., limiting rows to 129,934 if needed).

2. **Run the Notebook**:
   - Open `indian_text_summarization.ipynb` in Google Colab or a local Jupyter environment.
   - Execute cells sequentially to preprocess data, fine-tune the model, test, and launch the Gradio interface.

3. **Hugging Face Authentication**:
   - Obtain a Hugging Face API token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   - Store the token securely in Google Colab Secrets (`HF_TOKEN`) or as an environment variable:
     ```bash
     export HF_TOKEN=your_huggingface_token
     ```
   - **Security Note**: Avoid hardcoding tokens in the notebook to prevent accidental exposure.

4. **Launch the Gradio Interface**:
   - The final cell launches a Gradio interface for interactive summarization.
   - In Google Colab, the interface provides a public URL. For local environments, access it at `http://localhost:7860`.

## Project Structure
- `indian_text_summarization.ipynb`: The main Jupyter notebook containing the full pipeline (data loading, preprocessing, fine-tuning, testing, and Gradio interface).
- `/content/drive/My Drive/indicbartss_finetuned/`: Directory where the fine-tuned model and tokenizer are saved.

## Fine-Tuning IndicBARTSS
The model is fine-tuned with the following configuration:
- **Model**: `ai4bharat/IndicBARTSS`
- **Training Arguments**:
  - Learning Rate: `2e-5`
  - Per-Device Batch Size: `4` (with `gradient_accumulation_steps=2`)
  - Epochs: `3`
  - Weight Decay: `0.01`
  - Output Directory: `/content/drive/My Drive/indicbartss_finetuned`
  - Evaluation Strategy: Per epoch
  - Save Strategy: Per epoch, loading the best model at the end
- Training progress is logged to WandB for monitoring.
- The fine-tuned model and tokenizer are saved to Google Drive.

## Testing the Model
The model is tested on the test dataset, generating summaries for sample articles and comparing them to reference headlines. Example:
- **Article**: A Hindi news article.
- **Reference Summary**: The article’s headline.
- **Generated Summary**: The model’s output (may show repetition, see [Notes and Limitations](#notes-and-limitations)).

## Gradio Interface
The Gradio interface allows users to input a Hindi news article and receive a summary. Parameters:
- **Max Input Length**: 1024 tokens
- **Max Output Length**: 150 tokens
- **Min Output Length**: 50 tokens
- **Num Beams**: 4

To launch:
```python
interface.launch()
```

**Note**: In Colab, ensure the runtime is active to access the Gradio URL. Locally, check firewall settings if the interface is inaccessible.

## Dependencies
- `python==3.11`
- `transformers==4.38.2`
- `torch==2.0.0`
- `indic-nlp-library==0.92`
- `sentencepiece==0.2.0`
- `gradio==4.38.0`
- `datasets==2.21.0`
- `pandas==2.2.2`
- `wandb==0.19.11`

Install with:
```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with the above dependencies for reproducibility.

## Notes and Limitations
- **Dataset Availability**: Ensure you have access to `hindi_news_dataset.csv`. The notebook limits rows to 129,934 to handle potential dataset errors.
- **Model Output**: Generated summaries may exhibit repetition, possibly due to tokenization issues or insufficient fine-tuning. Consider adjusting `num_beams`, `max_length`, or increasing training epochs.
- **Gradio in Colab**: The Gradio interface may require a stable runtime. If the public URL fails, restart the Colab runtime or run locally.
- **WandB Setup**: Requires a WandB account and API key. Skip if not needed, but training logs won’t be saved.
- **Security**: Store API tokens (Hugging Face, WandB) securely using environment variables or Colab Secrets.
