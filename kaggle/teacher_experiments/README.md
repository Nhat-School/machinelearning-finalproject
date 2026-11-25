# Chọn notebook theo thí nghiệm trong DOCX

Đây là bộ notebook dành cho giảng viên. Mỗi file tự chứa mã và tải đúng dữ liệu; chỉ upload một file `.ipynb` tương ứng vào Kaggle.
Không cần chạy tất cả. Chọn đúng thí nghiệm được hỏi. Các notebook training tự huấn luyện khi Run All; bản 00 và 10 không train model.

| File | Nội dung | Trang DOCX | Thiết bị |
|---|---|---|---|
| [00_data_preparation_and_label_audit.ipynb](00_data_preparation_and_label_audit.ipynb) | Data preparation and the explicit AI label audit | 5–6, 16 | CPU |
| [01_tfidf_logistic_regression_and_dummy.ipynb](01_tfidf_logistic_regression_and_dummy.ipynb) | TF-IDF Logistic Regression and majority baseline | 7, 10 | CPU |
| [02_tfidf_svm.ipynb](02_tfidf_svm.ipynb) | TF-IDF linear SVM | 7, 10 | CPU |
| [03_tfidf_naive_bayes.ipynb](03_tfidf_naive_bayes.ipynb) | TF-IDF Multinomial Naive Bayes | 7, 10 | CPU |
| [04_frozen_minilm_logistic_regression.ipynb](04_frozen_minilm_logistic_regression.ipynb) | Frozen MiniLM embeddings with Logistic Regression | 7, 10 | GPU recommended |
| [05_source_minilm_finetuning_augmentation_cv.ipynb](05_source_minilm_finetuning_augmentation_cv.ipynb) | Source-specific MiniLM fine-tuning augmentation and five-fold CV | 8–9, 11 | GPU required |
| [06_minilm_tfidf_blending.ipynb](06_minilm_tfidf_blending.ipynb) | MiniLM and TF-IDF Logistic Regression blending | 8, 17 | GPU required |
| [07_joint_deberta_calibration_review.ipynb](07_joint_deberta_calibration_review.ipynb) | Joint DeBERTa calibration thresholds and review coverage | 8, 12, 18 | GPU required |
| [08_joint_minilm_two_datasets_selected.ipynb](08_joint_minilm_two_datasets_selected.ipynb) | Selected joint MiniLM on the two real datasets | 11, 13–18 | GPU required |
| [09_joint_minilm_three_dataset_comparison.ipynb](09_joint_minilm_three_dataset_comparison.ipynb) | Three-dataset MiniLM weak-supervision comparison | 13–18 | GPU required |
| [10_archived_logs_and_interrupted_attempt.ipynb](10_archived_logs_and_interrupted_attempt.ipynb) | Archived measured results and the interrupted FP32 attempt | 10–12; full comparison | CPU |

## Cách chạy

1. Chọn một file theo bảng; notebook 08 là model mặc định đã chọn.
2. Upload file vào Kaggle, bật Internet; dùng GPU T4 ×2 cho các bản neural. Mã dùng một GPU.
3. Run All. Không cần Add Input, API key hoặc notebook output cũ. Bản blend 06 tự train các model cần trước khi trộn.
4. Xem bảng chỉ số, ma trận, đường học/PR khi có, các lỗi thật và kiểm tra tái tạo. Đổi TEXT_TO_PREDICT để demo câu mới.
5. Tải ZIP output được in ở cell cuối. ZIP có model đã train, cấu hình, datasets/splits, code và mọi số đo của lần chạy.

## Phạm vi và các trường hợp dễ nhầm

- Bảng tham khảo trong phần giải thích là kết quả lịch sử. Kết quả mới nằm trong các cell chạy; không giả vờ số đo cũ là lần train mới.
- Các notebook mới kiểm tra phiên bản thư viện, khắc phục việc chỉ kiểm tra module đã tồn tại ở các notebook cũ. Python/GPU/PyTorch khác vẫn có thể làm số đo khác.
- Bản 09 dùng nhãn tổng hợp theo ngoại lệ synthetic weak supervision đã được yêu cầu. AI audit không phải human review; không tự duyệt worksheet.
- Bản 10 phát lại 83 dòng kết quả và hai log Kaggle đã cung cấp; nó không tái tạo weights từ các số đếm. Notebook khác dùng để train lại phương pháp.
- Lượt DeBERTa FP32 bị dừng trên Mac chưa có F1. Bản 10 giữ bằng chứng lỗi này; bản 07 chạy protocol DeBERTa hoàn tất. Không ép máy khác phải lỗi bộ nhớ giống Mac.
- Lượt 10 epoch chưa được thử nên không có notebook tuyên bố đây là thí nghiệm đã đo. Các cấu hình trong bộ này khớp những lượt đã báo cáo.
- Sản phẩm web ở trang 19–20 chạy local bằng bash Project/scripts/launchers/start_app.sh. Các notebook cung cấp cell demo dự đoán, không host web local.
- Bộ này không chứa model weights và không cài gì lên Mac khi chỉ tải/giải nén. Mọi mã/temp của quá trình tạo và kiểm tra nằm trong Project.

## Kiểm tra đã thực hiện

Xem VALIDATION.md và validation.json: phân biệt compile/bootstrap, đối soát dữ liệu, thực thi với kết quả training đã lưu và việc thực sự train trên Kaggle. Chưa tự nhận đã chạy GPU cho các notebook đóng gói mới.

Tạo lại bộ notebook: Project/.venv/bin/python Project/scripts/build_teacher_notebooks.py từ thư mục machine.

Đối chiếu từng dòng trong 83 kết quả: [REPORT_EXPERIMENT_MAP.md](REPORT_EXPERIMENT_MAP.md). Thư mục có kèm báo cáo Word chính thức 21 trang.
