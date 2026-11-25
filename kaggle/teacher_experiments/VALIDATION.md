# Validation of the teacher collection

All 11 notebooks were executed locally, cell by cell, in separate temporary workspaces. Completed model weights and saved predictions were reused; training was not repeated.
The checks cover embedded-source bootstrap, fresh data cleaning from locally copied pinned raw files, partition separation, recomputing statistics from predictions/confusion counts, figure generation, output packaging and inference demonstrations.
Fresh Kaggle package installation, network downloads and GPU training were not executed for this new collection. The training modules reuse the completed project implementations; numerical identity across hardware is not guaranteed.
Notebook outputs show LOCAL QA explicitly. On Kaggle, Run All defaults to fresh training in 01–09. Notebooks 00 and 10 do not train.

| Notebook | Code cells executed | Results checked | With per-post checks | Figures |
|---|---|---|---|---|
| 00_data_preparation_and_label_audit.ipynb | 15 | 0 | 0 | 1 |
| 01_tfidf_logistic_regression_and_dummy.ipynb | 18 | 8 | 8 | 3 |
| 02_tfidf_svm.ipynb | 18 | 4 | 4 | 3 |
| 03_tfidf_naive_bayes.ipynb | 18 | 4 | 4 | 3 |
| 04_frozen_minilm_logistic_regression.ipynb | 18 | 4 | 4 | 3 |
| 05_source_minilm_finetuning_augmentation_cv.ipynb | 19 | 8 | 8 | 4 |
| 06_minilm_tfidf_blending.ipynb | 23 | 4 | 4 | 3 |
| 07_joint_deberta_calibration_review.ipynb | 21 | 6 | 6 | 4 |
| 08_joint_minilm_two_datasets_selected.ipynb | 24 | 6 | 6 | 4 |
| 09_joint_minilm_three_dataset_comparison.ipynb | 24 | 15 | 15 | 7 |
| 10_archived_logs_and_interrupted_attempt.ipynb | 8 | 83 | 0 | 22 |
