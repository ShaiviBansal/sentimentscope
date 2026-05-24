# SentimentScope

Financial sentiment analysis system comparing a general-purpose baseline against a domain-adapted fine-tuned model.

## What it does
- Fine-tunes DistilBERT on financial news using Hugging Face PEFT/LoRA (updates <1% of parameters)
- Deploys two model versions behind a FastAPI inference API
- Validates improvement via statistically rigorous A/B testing (z-test)
- Monitors prediction drift in real time using Population Stability Index (PSI)

## Results
- Baseline accuracy: 18.3% (randomly initialized classification head)
- Fine-tuned accuracy: 81.8% (domain-adapted via LoRA)
- Training time: ~83 seconds on T4 GPU

## Tech Stack
- **Model:** DistilBERT + PEFT/LoRA (Hugging Face)
- **API:** FastAPI
- **Dashboard:** Streamlit
- **Training:** Google Colab (T4 GPU)
- **Model hosting:** Hugging Face Hub

## Run Locally

### API
```bash
cd SentimentScope
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --port 8000
```

### Dashboard
```bash
streamlit run dashboard/app.py
```