CÁC TỆP CHÍNH TRONG FINAL

FINAL_REPORT.docx: báo cáo chính, có giải thích thuật ngữ và mục lục.
FINAL_MODEL/: mô hình mặc định MiniLM, học từ UPenn và Stanford, seed 42.
FINAL_KAGGLE.ipynb: huấn luyện MiniLM hai nguồn trên Kaggle.
FINAL_DEMO_POSTS.csv: các bài đăng mẫu để thử nhập CSV trên ứng dụng.

CHẠY TRÊN MAC
FINAL_RUN_APP.command: nhấp đúp để mở ứng dụng web.
FINAL_RUN_MODEL.command: nhấp đúp để dự đoán trực tiếp bằng FINAL_MODEL.
Giữ FINAL bên trong Project vì hai lệnh dùng môi trường có sẵn của dự án.
Ứng dụng web dùng mô hình gốc trong Project/models; lệnh dự đoán trực tiếp dùng bản sao FINAL_MODEL.

CE VÀ PAIRWISE LÀ GÌ
CE: cách tính sai số giữa dự đoán và nhãn của từng câu khi huấn luyện.
Thêm CE tổng hợp: học thêm nhãn của các câu do trợ lý soạn.
Pairwise (học cặp): học để xếp câu còn cần hỗ trợ cao hơn câu đã giải quyết.
Đây là các cách huấn luyện, không phải tên tệp trong FINAL.
Mô hình mặc định chỉ học hai nguồn thật, không thêm dữ liệu tổng hợp hoặc TF-IDF.

CÁC BẢN SO SÁNH NẰM NGOÀI FINAL
Thêm ví dụ: ../models/urgency_final_minilm_custom_ce_seed42/
Học cặp: ../models/urgency_final_minilm_custom_pairwise_seed42/
Notebook so sánh: ../kaggle/final_minilm_trial.ipynb
Các câu tổng hợp và nhãn do trợ lý soạn, chưa được con người duyệt.

CHẠY TRÊN KAGGLE
Tải FINAL_KAGGLE.ipynb lên Kaggle, bật Internet và GPU, rồi Run All.
Notebook huấn luyện hai nguồn thật với seed 42 và 43, xuất mô hình seed 42.
Kết quả chạy mới có thể khác. FINAL_MODEL là trọng số đã lưu của thí nghiệm chính.

CÁC TỆP TXT HỖ TRỢ
requirements-ml.txt: danh sách thư viện cho học máy và ứng dụng.
requirements-ml.lock.txt: phiên bản thư viện để tái tạo môi trường.
requirements-notebook.txt: thư viện chạy notebook trên máy.
requirements-reliability.txt: thư viện bổ sung cho thí nghiệm DeBERTa.
requirements-report.txt: thư viện tạo báo cáo và hình ảnh.
SHA256SUMS.txt: mã kiểm tra tính toàn vẹn của tệp.
Giữ nguyên tên các tệp bên trong FINAL_MODEL để mô hình tải được.

NGUỒN BÁO CÁO
../reports/bao_cao_do_an_hoc_may.docx
Sau khi sinh lại báo cáo hoặc huấn luyện, cần đồng bộ bản sao trong FINAL.
Các script hỗ trợ nằm ở ../scripts/launchers/.
