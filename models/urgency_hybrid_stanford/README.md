# Offline MiniLM and TF-IDF blend

Copy this entire folder. Run: python predict.py --text "I cannot submit my assignment. Please help!"
Uses existing fine-tuned MiniLM and TF-IDF Logistic Regression. No encoder retraining.
Blend weight and threshold were selected on source OOF predictions, not test labels.
CPU FP32 scores may differ slightly from MPS BF16 evaluation.
Install requirements.txt in an isolated environment. scikit-learn is pinned for joblib compatibility.
