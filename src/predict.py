"""
Check messages from the command line:
  python src/predict.py "Congratulations! You won a free iPhone, call 09061701461 now"
  python src/predict.py "are we still meeting at 6?"
"""

import sys
from pathlib import Path

import joblib

bundle = joblib.load(Path(__file__).resolve().parents[1] / "models" / "spam_model.joblib")
model, threshold = bundle["model"], bundle["threshold"]

messages = sys.argv[1:] or [input("Message: ")]
for text, p in zip(messages, model.predict_proba(messages)[:, 1]):
    label = "SPAM" if p >= threshold else "ham"
    print(f"{label:<5} ({p:.0%} spam)  {text}")
