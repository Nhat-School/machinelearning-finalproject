# Joint DeBERTa urgency model

Trained on UPenn and Stanford fit partitions, with 0 approved custom training rows. Unreviewed drafts excluded.
Use: python predict.py --text "I cannot submit my assignment and the deadline is tonight."
Default mixed policy is shared across both sources. Source-specific policies are diagnostic alternatives.
REVIEW means a lecturer should judge the post. Calibration does not guarantee correctness.
Supported scope: English student forum posts. No validated semantic out-of-domain detector is included.
