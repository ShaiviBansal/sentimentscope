from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import numpy as np
from scipy import stats

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Label mapping
LABELS = {0: "Bearish", 1: "Bullish", 2: "Neutral"}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    ignore_mismatched_sizes=True
)

# Load fine-tuned model - replace with your HF username
import os
HF_USERNAME = "Shaivi"

HF_TOKEN = os.getenv("HF_TOKEN")
from huggingface_hub import login
if HF_TOKEN:
    login(token=HF_TOKEN)

finetuned_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    ignore_mismatched_sizes=True
)
finetuned_model = PeftModel.from_pretrained(
    finetuned_model,
    f"{HF_USERNAME}/sentimentscope-financial"
)
finetuned_model.eval()
base_model.eval()

# Store recent predictions for drift monitoring
recent_predictions = []

class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: list[str]
    labels: list[int]

def predict_single(model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().numpy()
    pred = int(np.argmax(probs))
    return {"label": LABELS[pred], "confidence": float(probs[pred]), "probabilities": {LABELS[i]: float(probs[i]) for i in range(3)}}

@app.get("/")
def root():
    return {"message": "SentimentScope API is running"}

@app.post("/predict")
def predict(input: TextInput):
    global recent_predictions
    base_result = predict_single(base_model, input.text)
    finetuned_result = predict_single(finetuned_model, input.text)
    recent_predictions.append(finetuned_result["label"])
    if len(recent_predictions) > 100:
        recent_predictions = recent_predictions[-100:]
    return {
        "text": input.text,
        "baseline": base_result,
        "finetuned": finetuned_result
    }

# @app.post("/ab_test")
# def ab_test(batch: BatchInput):
#     baseline_correct = 0
#     finetuned_correct = 0
#     n = len(batch.texts)

#     for text, label in zip(batch.texts, batch.labels):
#         base_pred = predict_single(base_model, text)
#         ft_pred = predict_single(finetuned_model, text)
#         if LABELS[label] == base_pred["label"]:
#             baseline_correct += 1
#         if LABELS[label] == ft_pred["label"]:
#             finetuned_correct += 1

#     base_acc = baseline_correct / n
#     ft_acc = finetuned_correct / n

#     # Two-proportion z-test
#     p1, p2 = base_acc, ft_acc
#     p_pool = (baseline_correct + finetuned_correct) / (2 * n)
#     se = np.sqrt(p_pool * (1 - p_pool) * (2 / n))
#     z_stat = float((p2 - p1) / se) if se > 0 else 0.0
#     p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

#     return {
#         "n_samples": int(n),
#         "baseline_accuracy": float(round(base_acc, 4)),
#         "finetuned_accuracy": float(round(ft_acc, 4)),
#         "improvement": float(round(ft_acc - base_acc, 4)),
#         "z_statistic": float(round(z_stat, 4)),
#         "p_value": float(round(p_value, 6)),
#         "significant": bool(p_value < 0.05)
#     }

# @app.get("/drift")
# def get_drift():
#     if len(recent_predictions) < 10:
#         return {"message": "Not enough predictions yet", "count": len(recent_predictions)}
    
#     counts = {label: recent_predictions.count(label) for label in LABELS.values()}
#     total = len(recent_predictions)
#     distribution = {k: round(v/total, 3) for k, v in counts.items()}
    
#     # Reference distribution from training data (approximate)
#     reference = {"Bearish": 0.33, "Bullish": 0.33, "Neutral": 0.34}
    
#     # Calculate PSI
#     psi = sum(
#         (distribution.get(k, 0.001) - reference[k]) * np.log((distribution.get(k, 0.001) + 0.001) / reference[k])
#         for k in reference
#     )
    
#     return {
#         "recent_distribution": distribution,
#         "reference_distribution": reference,
#         "psi": round(psi, 4),
#         "drift_detected": psi > 0.2,
#         "n_predictions": total
#     }


@app.post("/ab_test")
def ab_test(batch: BatchInput):
    baseline_correct = 0
    finetuned_correct = 0
    n = len(batch.texts)

    for text, label in zip(batch.texts, batch.labels):
        base_pred = predict_single(base_model, text)
        ft_pred = predict_single(finetuned_model, text)
        if LABELS[label] == base_pred["label"]:
            baseline_correct += 1
        if LABELS[label] == ft_pred["label"]:
            finetuned_correct += 1

    base_acc = baseline_correct / n
    ft_acc = finetuned_correct / n

    p_pool = (baseline_correct + finetuned_correct) / (2 * n)
    se = float(np.sqrt(p_pool * (1 - p_pool) * (2 / n)))
    z_stat = float((ft_acc - base_acc) / se) if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    return {
        "n_samples": n,
        "baseline_accuracy": round(base_acc, 4),
        "finetuned_accuracy": round(ft_acc, 4),
        "improvement": round(ft_acc - base_acc, 4),
        "z_statistic": round(z_stat, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05
    }

@app.get("/drift")
def get_drift():
    if len(recent_predictions) < 10:
        return {"message": "Not enough predictions yet", "count": len(recent_predictions)}

    counts = {label: recent_predictions.count(label) for label in LABELS.values()}
    total = len(recent_predictions)
    distribution = {k: round(v/total, 3) for k, v in counts.items()}
    reference = {"Bearish": 0.33, "Bullish": 0.33, "Neutral": 0.34}

    psi = float(sum(
        (distribution.get(k, 0.001) - reference[k]) * np.log((distribution.get(k, 0.001) + 0.001) / reference[k])
        for k in reference
    ))

    return {
        "recent_distribution": distribution,
        "reference_distribution": reference,
        "psi": round(psi, 4),
        "drift_detected": psi > 0.2,
        "n_predictions": total
    }