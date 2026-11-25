# Fine-tuned MiniLM urgency classifier

Source: upenn. English only.
Base revision: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41. Configuration: weighted_aug.
Full encoder and classification head trained for 3 epochs.
Load this directory locally with AutoTokenizer and AutoModelForSequenceClassification.
Apply decision_config.json threshold to softmax(logits)[:,1].
Previously inspected test sets are exploratory comparisons.
