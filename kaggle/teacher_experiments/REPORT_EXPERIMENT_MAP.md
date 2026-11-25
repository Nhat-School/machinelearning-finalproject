# Match every report result to a runnable notebook

Each row below identifies a recorded result. Upload the linked method notebook to train the method again; notebook 10 displays the recorded evidence without training.
A fresh run can differ from a historical score because hardware and software differ. Earlier Kaggle and Mac measurements remain separate.
Scouts, weighting, contraction augmentation and five-fold CV are all in 05 (and as prerequisites in 06). Both seeds and all completed custom ablations are in 09. Interrupted FP32/timing-only attempts are documented in 10, with no invented classifier score.

| Record | Original environment | Trial | Train → test | Policy / seed | F1 | Train again |
|---|---|---|---|---|---|---|
| 1 | Mac / original | TF-IDF + LR | upenn → upenn |  /  | 0.5299 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 2 | Mac / original | TF-IDF + LR | upenn → stanford |  /  | 0.4125 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 3 | Mac / original | TF-IDF + SVM | upenn → upenn |  /  | 0.5354 | [02](02_tfidf_svm.ipynb) |
| 4 | Mac / original | TF-IDF + SVM | upenn → stanford |  /  | 0.4420 | [02](02_tfidf_svm.ipynb) |
| 5 | Mac / original | TF-IDF + NB | upenn → upenn |  /  | 0.4052 | [03](03_tfidf_naive_bayes.ipynb) |
| 6 | Mac / original | TF-IDF + NB | upenn → stanford |  /  | 0.3937 | [03](03_tfidf_naive_bayes.ipynb) |
| 7 | Mac / original | Đa số | upenn → upenn |  /  | 0.0000 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 8 | Mac / original | Đa số | upenn → stanford |  /  | 0.0000 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 9 | Mac / original | TF-IDF + LR | stanford → upenn |  /  | 0.3929 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 10 | Mac / original | TF-IDF + LR | stanford → stanford |  /  | 0.7197 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 11 | Mac / original | TF-IDF + SVM | stanford → upenn |  /  | 0.3863 | [02](02_tfidf_svm.ipynb) |
| 12 | Mac / original | TF-IDF + SVM | stanford → stanford |  /  | 0.7184 | [02](02_tfidf_svm.ipynb) |
| 13 | Mac / original | TF-IDF + NB | stanford → upenn |  /  | 0.5000 | [03](03_tfidf_naive_bayes.ipynb) |
| 14 | Mac / original | TF-IDF + NB | stanford → stanford |  /  | 0.6432 | [03](03_tfidf_naive_bayes.ipynb) |
| 15 | Mac / original | Đa số | stanford → upenn |  /  | 0.0000 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 16 | Mac / original | Đa số | stanford → stanford |  /  | 0.0000 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 17 | Mac / original | MiniLM frozen + LR | upenn → upenn |  /  | 0.5414 | [04](04_frozen_minilm_logistic_regression.ipynb) |
| 18 | Mac / original | MiniLM frozen + LR | upenn → stanford |  /  | 0.4014 | [04](04_frozen_minilm_logistic_regression.ipynb) |
| 19 | Mac / original | MiniLM frozen + LR | stanford → upenn |  /  | 0.3920 | [04](04_frozen_minilm_logistic_regression.ipynb) |
| 20 | Mac / original | MiniLM frozen + LR | stanford → stanford |  /  | 0.6210 | [04](04_frozen_minilm_logistic_regression.ipynb) |
| 21 | Mac / extensions | MiniLM fine tuned | upenn → upenn |  /  | 0.6222 | [05](05_source_minilm_finetuning_augmentation_cv.ipynb) |
| 22 | Mac / extensions | MiniLM fine tuned | upenn → stanford |  /  | 0.6301 | [05](05_source_minilm_finetuning_augmentation_cv.ipynb) |
| 23 | Mac / extensions | MiniLM fine tuned | stanford → upenn |  /  | 0.4125 | [05](05_source_minilm_finetuning_augmentation_cv.ipynb) |
| 24 | Mac / extensions | MiniLM fine tuned | stanford → stanford |  /  | 0.8036 | [05](05_source_minilm_finetuning_augmentation_cv.ipynb) |
| 25 | Mac / extensions | MiniLM + TF-IDF | upenn → upenn |  /  | 0.6364 | [06](06_minilm_tfidf_blending.ipynb) |
| 26 | Mac / extensions | MiniLM + TF-IDF | upenn → stanford |  /  | 0.6268 | [06](06_minilm_tfidf_blending.ipynb) |
| 27 | Mac / extensions | MiniLM + TF-IDF | stanford → upenn |  /  | 0.4141 | [06](06_minilm_tfidf_blending.ipynb) |
| 28 | Mac / extensions | MiniLM + TF-IDF | stanford → stanford |  /  | 0.8092 | [06](06_minilm_tfidf_blending.ipynb) |
| 29 | Mac / extensions | DeBERTa joint / upenn | upenn+stanford → upenn | upenn /  | 0.5182 | [07](07_joint_deberta_calibration_review.ipynb) |
| 30 | Mac / extensions | DeBERTa joint / mixed | upenn+stanford → upenn | mixed /  | 0.5182 | [07](07_joint_deberta_calibration_review.ipynb) |
| 31 | Mac / extensions | DeBERTa joint / stanford | upenn+stanford → stanford | stanford /  | 0.7315 | [07](07_joint_deberta_calibration_review.ipynb) |
| 32 | Mac / extensions | DeBERTa joint / mixed | upenn+stanford → stanford | mixed /  | 0.7240 | [07](07_joint_deberta_calibration_review.ipynb) |
| 33 | Mac / extensions | DeBERTa joint / fixed 0.5 | joint → upenn | fixed 0.5 /  | 0.6119 | [07](07_joint_deberta_calibration_review.ipynb) |
| 34 | Mac / extensions | DeBERTa joint / fixed 0.5 | joint → stanford | fixed 0.5 /  | 0.7964 | [07](07_joint_deberta_calibration_review.ipynb) |
| 35 | Kaggle / supplied confusion counts | TF-IDF + LR | upenn → upenn |  /  | 0.5781 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 36 | Kaggle / supplied confusion counts | TF-IDF + SVM | upenn → upenn |  /  | 0.5289 | [02](02_tfidf_svm.ipynb) |
| 37 | Kaggle / supplied confusion counts | TF-IDF + NB | upenn → upenn |  /  | 0.3919 | [03](03_tfidf_naive_bayes.ipynb) |
| 38 | Kaggle / supplied confusion counts | Đa số | upenn → upenn |  /  | 0.0000 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 39 | Kaggle / supplied confusion counts | MiniLM frozen + LR | upenn → upenn |  /  | 0.5395 | [04](04_frozen_minilm_logistic_regression.ipynb) |
| 40 | Kaggle / supplied confusion counts | TF-IDF + LR | stanford → stanford |  /  | 0.7215 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 41 | Kaggle / supplied confusion counts | TF-IDF + SVM | stanford → stanford |  /  | 0.7185 | [02](02_tfidf_svm.ipynb) |
| 42 | Kaggle / supplied confusion counts | TF-IDF + NB | stanford → stanford |  /  | 0.6453 | [03](03_tfidf_naive_bayes.ipynb) |
| 43 | Kaggle / supplied confusion counts | Đa số | stanford → stanford |  /  | 0.0000 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 44 | Kaggle / supplied confusion counts | MiniLM frozen + LR | stanford → stanford |  /  | 0.6210 | [04](04_frozen_minilm_logistic_regression.ipynb) |
| 45 | Kaggle / supplied confusion counts | TF-IDF + LR | upenn → stanford |  /  | 0.4532 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 46 | Kaggle / supplied confusion counts | TF-IDF + SVM | upenn → stanford |  /  | 0.4259 | [02](02_tfidf_svm.ipynb) |
| 47 | Kaggle / supplied confusion counts | TF-IDF + NB | upenn → stanford |  /  | 0.3816 | [03](03_tfidf_naive_bayes.ipynb) |
| 48 | Kaggle / supplied confusion counts | Đa số | upenn → stanford |  /  | 0.0000 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 49 | Kaggle / supplied confusion counts | MiniLM frozen + LR | upenn → stanford |  /  | 0.4350 | [04](04_frozen_minilm_logistic_regression.ipynb) |
| 50 | Kaggle / supplied confusion counts | TF-IDF + LR | stanford → upenn |  /  | 0.3911 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 51 | Kaggle / supplied confusion counts | TF-IDF + SVM | stanford → upenn |  /  | 0.3882 | [02](02_tfidf_svm.ipynb) |
| 52 | Kaggle / supplied confusion counts | TF-IDF + NB | stanford → upenn |  /  | 0.5056 | [03](03_tfidf_naive_bayes.ipynb) |
| 53 | Kaggle / supplied confusion counts | Đa số | stanford → upenn |  /  | 0.0000 | [01](01_tfidf_logistic_regression_and_dummy.ipynb) |
| 54 | Kaggle / supplied confusion counts | MiniLM frozen + LR | stanford → upenn |  /  | 0.3920 | [04](04_frozen_minilm_logistic_regression.ipynb) |
| 55 | Kaggle / supplied FINAL TEST | MiniLM / selected | upenn → upenn | selected /  | 0.6087 | [05](05_source_minilm_finetuning_augmentation_cv.ipynb) |
| 56 | Kaggle / supplied FINAL TEST | MiniLM / fixed 0.5 | upenn → upenn | fixed 0.5 /  | 0.5917 | [05](05_source_minilm_finetuning_augmentation_cv.ipynb) |
| 57 | Kaggle / supplied FINAL TEST | MiniLM / selected | upenn → stanford | selected /  | 0.6387 | [05](05_source_minilm_finetuning_augmentation_cv.ipynb) |
| 58 | Kaggle / supplied FINAL TEST | MiniLM / fixed 0.5 | upenn → stanford | fixed 0.5 /  | 0.6655 | [05](05_source_minilm_finetuning_augmentation_cv.ipynb) |
| 59 | Kaggle / supplied FINAL TEST | MiniLM / selected | stanford → upenn | selected /  | 0.4032 | [05](05_source_minilm_finetuning_augmentation_cv.ipynb) |
| 60 | Kaggle / supplied FINAL TEST | MiniLM / fixed 0.5 | stanford → upenn | fixed 0.5 /  | 0.3883 | [05](05_source_minilm_finetuning_augmentation_cv.ipynb) |
| 61 | Kaggle / supplied FINAL TEST | MiniLM / selected | stanford → stanford | selected /  | 0.7997 | [05](05_source_minilm_finetuning_augmentation_cv.ipynb) |
| 62 | Kaggle / supplied FINAL TEST | MiniLM / fixed 0.5 | stanford → stanford | fixed 0.5 /  | 0.7843 | [05](05_source_minilm_finetuning_augmentation_cv.ipynb) |
| 63 | Kaggle / supplied joint MiniLM log 2026-11-17 | Joint MiniLM / s42 | joint → upenn |  / 42 | 0.6613 | [08](08_joint_minilm_two_datasets_selected.ipynb) |
| 64 | Kaggle / supplied joint MiniLM log 2026-11-17 | Joint MiniLM / s42 | joint → stanford |  / 42 | 0.7701 | [08](08_joint_minilm_two_datasets_selected.ipynb) |
| 65 | Kaggle / supplied joint MiniLM log 2026-11-17 | Joint MiniLM / s43 | joint → upenn |  / 43 | 0.6260 | [08](08_joint_minilm_two_datasets_selected.ipynb) |
| 66 | Kaggle / supplied joint MiniLM log 2026-11-17 | Joint MiniLM / s43 | joint → stanford |  / 43 | 0.7605 | [08](08_joint_minilm_two_datasets_selected.ipynb) |
| 67 | Kaggle / supplied joint MiniLM log 2026-11-17 | Joint MiniLM / s42 | joint → custom_challenge |  / 42 | 0.8000 | [08](08_joint_minilm_two_datasets_selected.ipynb) |
| 68 | Kaggle / supplied joint MiniLM log 2026-11-17 | Joint MiniLM / s43 | joint → custom_challenge |  / 43 | 0.8000 | [08](08_joint_minilm_two_datasets_selected.ipynb) |
| 69 | Mac / final joint | B real-only / 42 | joint → upenn |  / 42 | 0.6667 | [08](08_joint_minilm_two_datasets_selected.ipynb) |
| 70 | Mac / final joint | B real-only / 42 | joint → stanford |  / 42 | 0.7742 | [08](08_joint_minilm_two_datasets_selected.ipynb) |
| 71 | Mac / final joint | C custom CE / 42 | joint → upenn |  / 42 | 0.6667 | [09](09_joint_minilm_three_dataset_comparison.ipynb) |
| 72 | Mac / final joint | C custom CE / 42 | joint → stanford |  / 42 | 0.7598 | [09](09_joint_minilm_three_dataset_comparison.ipynb) |
| 73 | Mac / final joint | C custom pair / 42 | joint → upenn |  / 42 | 0.6612 | [09](09_joint_minilm_three_dataset_comparison.ipynb) |
| 74 | Mac / final joint | C custom pair / 42 | joint → stanford |  / 42 | 0.7607 | [09](09_joint_minilm_three_dataset_comparison.ipynb) |
| 75 | Mac / final joint | B real-only / 43 | joint → upenn |  / 43 | 0.6512 | [08](08_joint_minilm_two_datasets_selected.ipynb) |
| 76 | Mac / final joint | B real-only / 43 | joint → stanford |  / 43 | 0.7661 | [08](08_joint_minilm_two_datasets_selected.ipynb) |
| 77 | Mac / final joint | C custom pair / 43 | joint → upenn |  / 43 | 0.6179 | [09](09_joint_minilm_three_dataset_comparison.ipynb) |
| 78 | Mac / final joint | C custom pair / 43 | joint → stanford |  / 43 | 0.7630 | [09](09_joint_minilm_three_dataset_comparison.ipynb) |
| 79 | Mac / synthetic diagnostic | B real-only / 42 | joint → custom_challenge |  / 42 | 0.8000 | [08](08_joint_minilm_two_datasets_selected.ipynb) |
| 80 | Mac / synthetic diagnostic | C custom CE / 42 | joint → custom_challenge |  / 42 | 1.0000 | [09](09_joint_minilm_three_dataset_comparison.ipynb) |
| 81 | Mac / synthetic diagnostic | C custom pair / 42 | joint → custom_challenge |  / 42 | 1.0000 | [09](09_joint_minilm_three_dataset_comparison.ipynb) |
| 82 | Mac / synthetic diagnostic | B real-only / 43 | joint → custom_challenge |  / 43 | 0.8000 | [08](08_joint_minilm_two_datasets_selected.ipynb) |
| 83 | Mac / synthetic diagnostic | C custom pair / 43 | joint → custom_challenge |  / 43 | 1.0000 | [09](09_joint_minilm_three_dataset_comparison.ipynb) |
