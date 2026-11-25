# Offline urgency model

Copy this entire folder to another project.
Run: python predict.py --text "I cannot submit my assignment. Please help!"
Python needs torch and transformers; no Hugging Face cache, token or Internet is required for inference.
The exact text cleaning function is included in predict.py. Labels and threshold are in decision_config.json.
Training/validation used the project protocol. FP32 CPU inference can differ slightly from bfloat16 evaluation.
