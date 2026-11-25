# Forum Priority

Web phân loại và ưu tiên phản hồi bài đăng tiếng Anh, kèm báo cáo Word và thí nghiệm chạy lại.
Toàn bộ thư viện mới, dữ liệu, mô hình và cache nằm trong `Project`; các slide ở thư mục cha giữ nguyên.

## Đồng đội tải và chạy dự án

Repository chứa mã nguồn, dữ liệu, báo cáo, kết quả thí nghiệm và các model đã huấn luyện. Không cần train lại để chạy ứng dụng. Cần Git LFS để tải đủ trọng số; `.venv` và cache được tạo riêng trên từng máy.

1. Cài Git LFS và Python 3.12 trên máy của bạn. Chạy `git lfs install` một lần.
2. Tải repository vào thư mục tên `Project` để khớp các ví dụ lệnh trong báo cáo:

   ```bash
   git clone https://github.com/Nhat-School/machinelearning-finalproject.git Project
   cd Project
   git lfs pull
   git lfs fsck
   ```

3. Cài môi trường trong thư mục vừa tải:

   ```bash
   bash scripts/launchers/setup.sh
   ```

4. Chạy ứng dụng:

   ```bash
   bash scripts/launchers/start_app.sh
   ```

   Mở http://127.0.0.1:8765 và giữ Terminal đang chạy. Thử nhập `FINAL/FINAL_DEMO_POSTS.csv` ở màn hàng đợi. Model mặc định là MiniLM học từ UPenn và Stanford, seed 42.

Báo cáo để đọc: [FINAL_REPORT.docx](FINAL/FINAL_REPORT.docx). Notebook train mặc định trên Kaggle: [FINAL_KAGGLE.ipynb](FINAL/FINAL_KAGGLE.ipynb). Các notebook so sánh nằm trong `kaggle/`.

Không dùng nút Download ZIP của GitHub để tải trọng số: hãy clone và chạy `git lfs pull`. Nếu repository riêng tư, tài khoản của đồng đội cần được cấp quyền truy cập. Lệnh ở trên dành cho macOS/Linux; trên Windows có thể dùng WSL.

Chạy cùng model cho phép dùng lại ứng dụng; huấn luyện mới trên phần cứng hoặc phiên bản thư viện khác có thể cho điểm khác. Danh sách phiên bản đã lưu nằm trong `requirements-ml.lock.txt`.

## Bản nộp cuối: notebook chính và báo cáo chính thức

- **Bộ chạy từng thí nghiệm cho giảng viên:** [teacher_experiments](kaggle/teacher_experiments/), gồm 11 notebook đánh số 00–10. [Bảng chọn file theo trang DOCX](kaggle/teacher_experiments/README.md). Upload một notebook lên Kaggle, Internet ON và Run All; mỗi bản tự chuẩn bị dữ liệu và các model cần trước. Bản 08 là prototype đã chọn; 09 là so sánh ba nguồn; 00/10 không train. Mọi cell đã kiểm tra local với training đã lưu; không khẳng định bộ mới đã chạy GPU Kaggle.
- **Notebook cho giảng viên:** `kaggle/final_best_model.ipynb`, train MiniLM joint hai nguồn thật, không dùng custom để cập nhật trọng số. Internet ON, T4 ×2, Run All; không cần thêm dataset/API key.
- **Báo cáo chính:** `reports/bao_cao_do_an_hoc_may.docx`, 21 trang, 83 dòng so sánh, giải thích thắng / thua và khác biệt hai notebook MiniLM, các thử nghiệm thất bại, ma trận và hướng dẫn web ngắn ở trang 19–20.
- **Model mặc định:** `models/urgency_final_minilm_real_only_seed42`. Chạy app bằng `bash Project/scripts/launchers/start_app.sh`.
- **Kết quả weights Mac trong app:** F1 UPenn 0.6667, Stanford 0.7742; đúng 14/16 probe binary draft. Không phải model cao F1 nhất trên mọi nguồn: blend Stanford vẫn cao hơn.
- **Log Kaggle joint mới do người dùng cung cấp:** seed 42 có F1 UPenn 0.6613, Stanford 0.7701; đúng 12/16 probe. Lưu riêng trong `reports/kaggle_run/joint_minilm_20261117_*`; chưa nhập weights hoặc CSV từng bài của Kaggle vào app.
- **Kiểm tra nhãn custom bằng AI:** `Datasets/CustomForum/ai_label_audit_20261117.csv`, 160 dòng. Giữ rõ reviewer là AI, không human approval; 40 cập nhật still-blocked trùng hậu tố. Tái tạo: `Project/.venv/bin/python Project/scripts/audit_custom_labels_ai.py` từ machine.
- **Kết luận custom:** nhánh ba nguồn không thắng đối chứng trên hai test và hai seed; giữ trong notebook `final_minilm_trial.ipynb` để đối chiếu, không nâng thành model mặc định.
- Tái chạy model cuối: `bash Project/scripts/launchers/reproduce.sh --best-minilm`. Tái chạy mọi nhánh cuối: `bash Project/scripts/launchers/reproduce.sh --final-minilm`. Chỉ tạo lại bằng chứng: `bash Project/scripts/launchers/reproduce.sh`.
- Model cuối nằm trong `models/urgency_final_minilm_real_only_seed42`; thống kê trong `reports/final_minilm_trial`. Các ZIP đã được dọn; bản validation cũ được giữ trong `reports/backups/zip_versions`. Notebook chính đã kiểm tra local 23 code cell bằng training đã lưu. Phiên Kaggle trong log người dùng là lần chạy riêng; bảng lịch sử ngay đầu notebook giải thích các phương pháp đã thử. Giới hạn 10 epoch chưa được thử; không tự lặp thí nghiệm cũ để bổ sung ghi chú.

## Thực nghiệm DeBERTa và cơ chế xem lại

Phần mở rộng dùng **microsoft/deberta-v3-base**, 512 token, huấn luyện chung UPenn và Stanford.
Lượt này **chưa chứng minh tốt hơn các model cũ trên cả hai nguồn**. Policy chung theo validation balanced accuracy
đạt Recall 0.9344 / 0.9529 nhưng F1 0.5182 / 0.7240 (UPenn / Stanford) vì nhiều báo nhầm.
Chẩn đoán ngưỡng cố định 0.5 đạt F1 0.6119 / 0.7964, không thay policy đã chọn.
Trên 40 cặp câu nháp có cập nhật đã giải quyết, điểm giảm ở 29 cặp với DeBERTa và 34 cặp với MiniLM;
đây là khảo sát hành vi, chưa phải accuracy với nhãn độc lập. Ví dụ lỗi nộp bài vẫn bị gán ưu tiên sau khi đã báo sửa xong.
Vì vậy giữ DeBERTa như ứng viên thực nghiệm và giữ các model cũ để so sánh; không gọi đây là bản nâng cấp đã được xác nhận.

Mục tiêu là quan sát lỗi thực tế và chuyển các bài chưa chắc chắn sang **REVIEW** để giảng viên đọc.
Điểm cao và calibration vẫn không bảo đảm đúng mọi câu. Phạm vi đánh giá là bài đăng học tập **tiếng Anh**;
chưa có bộ phát hiện ngữ nghĩa ngoài phạm vi được kiểm định.

Dữ liệu nay cùng nằm ở **Datasets/**:

- `UPenn/raw.csv`, `UPenn/processed.csv`: nguồn nghiên cứu gốc, giữ nguyên nhãn.
- `Stanford/raw.csv`, `Stanford/processed.csv`: nguồn nghiên cứu gốc, giữ nguyên nhãn.
- `CustomForum/draft_examples.csv`: **160 câu tổng hợp do trợ lý soạn**, 80 nhóm tình huống, chưa phải nhãn gold.
- `CustomForum/reviewed_labels.csv`: người đọc duyệt nhãn theo `ANNOTATION_GUIDE.md`; không bị ghi đè khi chạy lại.

Các đường dẫn cũ dưới data/ là liên kết tương đối đến Datasets, không nhân đôi hai bộ dữ liệu thật.
Không tự sửa nhãn hoặc thêm câu tạo sinh dưới tên UPenn/Stanford. CustomForum có cặp vấn đề chưa giải quyết và đã giải quyết,
bài thiếu ngữ cảnh, trích dẫn và nội dung ngoài phạm vi. Mặc định **không dùng nhãn nháp để train**.
Chỉ các hàng train đã có reviewer và nhãn adjudicated mới được nhận khi bật `--include-approved-custom`.
Các nhóm validation/challenge giữ riêng. Chưa có số đo accuracy độc lập trên custom drafts.

Quy trình DeBERTa tách fit, calibration, validation và test. UPenn giữ student group giữa mọi phần chia.
Checkpoint theo AP validation trung bình hai nguồn; ngưỡng theo balanced accuracy validation.
Policy chung dùng calibration cân bằng nguồn; với ngưỡng đã chọn, điểm ≤ 0.07 hoặc ≥ 0.85 mới tự quyết định.
Các điểm giữa chuyển REVIEW. Số liệu binary vẫn tính trên **tất cả** bài test, đồng thời ghi coverage, review và lỗi tự động.
Test lịch sử đã được xem; đây là so sánh exploratory, chưa có fresh external test.

```bash
# Huấn luyện thật DeBERTa trên máy và cập nhật cùng Word
bash Project/scripts/launchers/reproduce.sh --reliable

# Tạo lại bằng chứng từ model và kết quả đã lưu, không train lại
bash Project/scripts/launchers/reproduce.sh

# Dùng model folder độc lập, không cần Internet
bash -c 'source Project/scripts/launchers/env.sh; Project/.venv/bin/python Project/models/urgency_deberta_real_only/predict.py --text "I cannot submit my assignment. Please help!"'
```

Trong app chọn **UPenn + Stanford · DeBERTa joint + review** sau khi model đã hoàn tất và kiểm tra offline.
CSV xuất có cột `action`; REVIEW chưa được coi là đã phân loại tự động, dù vẫn có gợi ý nhị phân.
Model cũ giữ để so sánh. `reports/reliability/deberta_real_only` chứa protocol, history, predictions, metrics,
audit độ tin cậy và khảo sát hành vi trên câu nháp. Lần FP32 dừng vì áp lực bộ nhớ giữ trong `fp32_runtime_attempt.json`;
không có số đo chất lượng cho lượt đã dừng. Lượt Mac dùng BF16 và batch 4 để giảm bộ nhớ.

**Notebook mới:** `kaggle/reliable_deberta_training.ipynb`. Bật Internet và GPU T4 ×2 rồi Run All;
trainer dùng **một T4 với FP16**, tự tải hai nguồn thật và tạo cả ba thư mục dữ liệu. Không cần upload thêm dataset cho real-only run.
Notebook hiển thị metric, ma trận, loss, calibration, coverage/review, lỗi và câu dự đoán; xuất ZIP model/dữ liệu/thống kê.
Muốn train custom đã duyệt, thêm worksheet vào input và chép đúng `Project/Datasets/CustomForum/reviewed_labels.csv`
**trước** bước chuẩn bị, rồi đặt `USE_APPROVED_CUSTOM=True`. Run này lưu riêng `deberta_reviewed_custom`.
Không dùng draft challenge làm test xác nhận độc lập. Đánh giá model trên bài thật mới đã gán nhãn vẫn là bước cần làm.

## Vì sao model lớn hơn không luôn thắng và MiniLM ba nguồn

Báo cáo chính có phần so sánh riêng MiniLM và DeBERTa, gồm kích thước checkpoint, ngưỡng đã chọn,
đối chiếu thêm ngưỡng cố định 0.5, các lỗi trạng thái đã giải quyết và giới hạn của phép so sánh.
Số liệu được tạo lại bằng `scripts/summarize_model_comparison.py`; `reproduce.sh` gọi tự động.
Cùng ngưỡng số vẫn chưa đồng nhất calibration và thành phần train, nên chưa phải thí nghiệm cô lập dung lượng model.

Nhánh MiniLM cuối có trainer riêng cho joint hai nguồn và custom CE / custom pairwise.
Thí nghiệm được yêu cầu cho phép dùng **82 nhãn custom dự thảo như synthetic weak supervision**, giữ rõ provenance;
không biến các nhãn này thành human approved. Validation có 30 và challenge có 24 hàng binary, tách theo nhóm tình huống.
So sánh hai nguồn thật, lỗi cập nhật và hai seed; xem `reports/final_minilm_trial/summary.json` để biết trạng thái và số đo thật.
Cả hai nhánh khởi tạo từ cùng pretrained MiniLM. Checkpoint fine-tuned cũ đã thấy các hàng train cũ nay được tách thành
calibration/validation cho joint, nên tiếp tục checkpoint cũ cần development data mới hoặc thiết kế lại đánh giá.
Giữ riêng các model cũ; không dùng test để quyết định cấu hình mới.

Notebook cho lượt cuối là **`kaggle/final_minilm_trial.ipynb`**. Bật Internet và T4 ×2, Run All;
không cần thêm dataset/API key. `RUN_SYNTHETIC_TRIAL=True` là lựa chọn tường minh cho thí nghiệm nhãn yếu,
không phải chứng nhận nhãn. Notebook in metric, confusion matrix, loss, hai seed, lỗi và các tình huống;
cho nhập câu dự đoán và xuất `final_minilm_models_and_statistics.zip`.
Kết quả custom là chẩn đoán trên dữ liệu tổng hợp, chưa phải accuracy ngoài thực tế. Model dùng một T4, FP16;
trên Mac dùng MPS BF16. Local notebook validation tái dùng training thật, không khẳng định đã chạy notebook mới trên Kaggle GPU.

```bash
# Tái chạy lượt cuối; các run đã hoàn tất được giữ lại
bash Project/scripts/launchers/reproduce.sh --final-minilm

# Chỉ đối chứng hai nguồn, không dùng draft để cập nhật trọng số
bash -c 'source Project/scripts/launchers/env.sh; Project/.venv/bin/python Project/scripts/run_final_minilm_trial.py --bf16'

# Phân loại từ model joint hai nguồn độc lập, không cần server
bash -c 'source Project/scripts/launchers/env.sh; Project/.venv/bin/python Project/models/urgency_final_minilm_real_only_seed42/predict.py --text "I cannot submit my assignment. Please help!"'
```

Trong web chọn **MiniLM joint 2 nguồn · seed 42** hoặc **MiniLM + CustomForum · thử nghiệm**.
Các model cũ được giữ cho phần so sánh; model mặc định mới là joint real_only seed42. `selection.json` ghi cấu hình custom chọn bằng validation;
`frozen_recommendation.json` ghi kết quả xác nhận ở seed 43 trước khi đọc test custom.

## Tối ưu từ các mô hình đã train

Model trộn có sẵn trong `models/urgency_hybrid_stanford` và `models/urgency_hybrid_upenn`.
Trong app chọn **MiniLM + TF-IDF blend**. Trọng số và ngưỡng chọn bằng OOF train nguồn, không bằng test.
Phiên Mac: Stanford F1 tăng 0.803564 → 0.809199; UPenn 0.622222 → 0.636364.
Chuyển UPenn → Stanford giảm nhẹ; không bảo đảm luôn cải thiện hoặc áp dụng cùng số cho phiên Kaggle.

`kaggle/optimize_existing_models.ipynb` chỉ dùng các kết quả cũ, không train lại encoder.
Trong session Kaggle cũ còn Project: Run All. Trong session mới: Add Input output của notebook trước
(cần reports/experiments, reports/fine_tuning và models). ZIP cũ 171.7 MB không có OOF của LR nên chưa đủ.
Thư mục hybrid độc lập chứa cả hai model và `predict.py`; đã kiểm tra CPU offline.

```bash
bash -c 'source Project/scripts/launchers/env.sh; Project/.venv/bin/python Project/models/urgency_hybrid_stanford/predict.py --text "I cannot submit my assignment. Please help!"'
```

## MiniLM fine tuned và notebook Kaggle đầy đủ

Hai model độc lập: `models/urgency_minilm_upenn` và `models/urgency_minilm_stanford`.
Mỗi folder chứa model, tokenizer, threshold và `predict.py` tự chứa; có thể copy cả folder sang project khác.
Trong web, chọn `MiniLM fine-tuned` ở dropdown. Mặc định TF-IDF + LR cũ được giữ để so sánh.

```bash
# Dự đoán offline từ một model folder
bash -c 'source Project/scripts/launchers/env.sh; Project/.venv/bin/python Project/models/urgency_minilm_stanford/predict.py --text "I cannot submit my assignment. Please help!"'

# Chạy lại 3 cấu hình scout, 5-fold, fit cuối và cập nhật cùng DOCX
bash Project/scripts/launchers/reproduce.sh --finetune
```

Fine tuning cập nhật toàn bộ encoder và head. Thử plain loss, weighted loss và weighted loss cộng
contraction expansion; chọn riêng bằng validation của mỗi nguồn. Giữ toàn bộ lượt kém cùng lịch sử.
Chạy tối đa 3 epoch, patience 1 theo AP validation; ngưỡng dựa trên OOF train. Test đã được xem trước,
nên các so sánh mở rộng là exploratory. Điểm CV có dùng fold validation cho early stopping, không phải nested CV.

`kaggle/urgency_minilm_training.ipynb` là notebook self-contained để upload vào Kaggle.
Bật Internet và GPU rồi Run All. Notebook mặc định train lại và hiển thị các bảng, hình, lỗi và dự đoán câu mới.
GPU T4 ×2 dùng được; trainer hiện dùng một GPU, chưa phân tán trên cả hai. Trên T4 dùng FP32,
chỉ bật BF16 trên CUDA có hỗ trợ phần cứng trực tiếp để tránh chọn nhầm chế độ emulation.
Bản đã chạy local ghi rõ QA tái dùng training đã hoàn thành; không giả lập rằng đã chạy trên Kaggle GPU.
Tạo lại notebook và kiểm tra các cell bằng:

```bash
bash -c 'source Project/scripts/launchers/env.sh; Project/.venv/bin/python Project/scripts/build_kaggle_notebook.py; Project/.venv/bin/python Project/scripts/execute_kaggle_notebook.py'
```

Notebook đầy đủ hiện có 28 kết quả (20 baseline/frozen, 4 fine-tuned và 4 hybrid) cùng ZIP xuất model/thống kê.
Tổng thời gian fine tuning lâu hơn pipeline frozen; xem thời gian từng lượt trong `reports/fine_tuning`.

## Phiên Kaggle do người dùng cung cấp

`reports/kaggle_run` giữ riêng bốn dòng FINAL TEST, môi trường và kết quả từ log Kaggle người dùng đã gửi.
Stanford → Stanford đạt F1 0.799659; UPenn → Stanford đạt 0.638743. Hai giá trị thuộc hai model khác nhau.
Các bảng, ma trận và hình so sánh được thêm vào cùng báo cáo Word. Trọng số app local chưa thay bằng model Kaggle.
Tạo lại hình và kiểm tra số đếm bằng `Project/.venv/bin/python Project/scripts/summarize_kaggle_run.py`
sau khi source `Project/scripts/launchers/env.sh`, hoặc chạy `bash Project/scripts/launchers/reproduce.sh`.

Log cho thấy embeddings frozen từng chạy CPU trên Kaggle; notebook hiện đã sửa để chọn CUDA khi có.
Đây là thay đổi cho lần chạy sau, không phải thay đổi kết quả của phiên đã ghi. Không cần chạy lại phiên đã thành công
chỉ để xem lại thống kê. Phiên bản thư viện Kaggle khác Mac; giữ notebook output và ZIP khi nộp.
Các dependencies notebook được ghi riêng trong `requirements-notebook.txt`; tất cả cài vào `.venv`.

## Mở ứng dụng từ machine

```bash
bash Project/scripts/launchers/start_app.sh
```

Mở **http://127.0.0.1:8765**. Nhập câu tự do hoặc dùng ví dụ, chọn mô hình để so sánh.
Hoặc nhấp đúp `Project/MoUngDung.command` trong Finder để bật server và mở trình duyệt.
Hàng đợi nhận CSV có cột `text`, `post_text` hoặc `text_clean`, tối đa 500 bài; xuất CSV kết quả.
Trang thực nghiệm đọc kết quả huấn luyện, không dùng số demo. Ctrl+C để dừng server.
Giao diện tiếng Việt; phạm vi huấn luyện và đánh giá là **tiếng Anh**. Không gọi API AI bên ngoài.

## Báo cáo chính và tái tạo

`reports/bao_cao_do_an_hoc_may.docx` là báo cáo đầy đủ. Bản `minh_chung_xu_ly_du_lieu.docx` chỉ ghi bước chuẩn bị trước đây.

```bash
# Tạo lại biểu đồ, ảnh Chrome và Word từ kết quả đã lưu
bash Project/scripts/launchers/reproduce.sh

# Huấn luyện lại toàn bộ rồi tạo lại minh chứng
bash Project/scripts/launchers/reproduce.sh --train

# Kiểm tra tính toàn vẹn tệp
python3 Project/scripts/verify_evidence.py

# Refit mô hình, so split và 20 bộ chỉ số với lần trước
python3 Project/scripts/check_ml_reproducibility.py
```

Kiểm tra ML yêu cầu cùng split, cùng ma trận nhầm lẫn và chênh chỉ số dưới 1e-8; bỏ qua thời gian chạy.
Sau kiểm tra ML, chạy `reproduce.sh` để cập nhật Word và manifest. Embeddings cache được tái dùng
nếu hash CSV và revision MiniLM khớp. Không huấn luyện lại encoder pretrained.
Kết quả là một seed hold-out; không suy ra ý nghĩa thống kê của chênh lệch mô hình.

## Môi trường riêng

`.venv` dùng Python 3.12 đã có, không kế thừa thư viện global. Các gói ML chỉ cài tại đây.
`env.sh` đặt cache pip/Hugging Face/Torch/matplotlib và file tạm vào `.cache` trong Project.
Không sửa `.zshrc`, cài Homebrew hoặc chỉnh Python global. Bộ runtime Word và Playwright có sẵn
của Codex được dùng chỉ đọc. Mã chụp ảnh dùng Google Chrome đã có, không tải Chromium mới.

Tạo lại môi trường sau khi xóa `.venv` bằng phiên bản đã khóa:

```bash
cd Project
bash -c 'source scripts/launchers/env.sh; python3.12 -m venv .venv; .venv/bin/python -m pip install --only-binary=:all: -r requirements-ml.lock.txt'
```

`setup.sh` dùng khi khởi tạo lưới phiên bản và xuất lock; ưu tiên lock để tái lập môi trường.
Tạo Word/chụp ảnh dùng runtime Codex đã có trong `~/.cache/codex-runtimes/codex-primary-runtime`.
Khi chuyển sang máy khác cần Python có python-docx/Pillow, Node/Playwright và đường dẫn Chrome phù hợp.

## Thiết kế thí nghiệm

- Nhãn: `priority = int(urgency > 4)`, không làm tròn nửa điểm Stanford.
- UPenn: hold-out theo student ID và CV ba fold giữ nhóm.
- Stanford: stratified hold-out và CV ba fold; bản CSV không có ID tác giả/luồng.
- TF-IDF unigram/bigram, min_df 2, max_df .98, tối đa 40000 đặc trưng, sublinear TF.
- LR/SVM/NB, Dummy đa số và phần mở rộng MiniLM 384 chiều + LR.
- Tuning bằng F1 trên train nguồn; ngưỡng chọn từ OOF nguồn. Test đích không dùng tuning.
- Cùng split cho mọi phương pháp. Bốn hướng đánh giá, tổng 20 bộ kết quả test.
- Chọn mô hình web theo F1 OOF trong nhóm Stanford, không chọn theo test.

## Vị trí bằng chứng

- `Datasets/UPenn`, `Datasets/Stanford`, `Datasets/CustomForum`: dữ liệu tập trung, provenance và worksheet duyệt nhãn; data/raw và data/processed là liên kết tương thích.
- `reports/data_audit.json`, `rejected_rows.csv`: kiểm tra chất lượng và lý do loại từng dòng.
- `reports/experiments/config.json`, `split_ids.csv`: cấu hình và tập chia.
- `cv_results.json`, `results.json` trong experiments: tuning và test.
- `predictions_*.csv`, `oof_*.csv`: từng nhãn và điểm để kiểm tra và phân tích lỗi.
- `encoder_info.json`: revision, token bị cắt, thiết bị và thời gian mã hóa.
- `reports/figures`, `screenshots`: biểu đồ và ảnh Chrome thật.
- `screenshots/browser_checks.json`: đối soát API/UI, hàng đợi, CSV, input rỗng, mobile.
- `models/*.joblib`, `models/deployment.json`: mô hình và quyết định triển khai.
- `reports/evidence_manifest.json`: hash mã và bằng chứng, không gồm cache, venv hay file khóa Office.

Xem dung lượng bằng `du -sh Project`. Dừng server rồi xóa `Project` để gỡ thành phần dự án.
Giữ runtime Codex dùng chung. Không tự phát hành lại dữ liệu lên Kaggle.

## Nguồn

https://github.com/pcla-code/forum-posts-urgency

https://arxiv.org/abs/2307.07614

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

https://scikit-learn.org/stable/
