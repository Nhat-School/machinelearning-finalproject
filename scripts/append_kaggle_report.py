"""Append the user-supplied Kaggle run as separate evidence in the existing report."""
import json
from pathlib import Path
from docx.shared import Pt
from build_report import table,figure

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';K=R/'kaggle_run'
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def fmt(value):return f'{value:.4f}'

def append_kaggle_run(document):
    if not (K/'results.json').exists():return
    rows=load(K/'results.json');environment=load(K/'environment.json')
    local=load(R/'fine_tuning/results.json')
    def p(text):document.add_paragraph(text)
    def page(title):document.add_page_break();document.add_heading(title,1)
    def h(text):document.add_heading(text,2)
    def code(text):
        para=document.add_paragraph(text)
        for run in para.runs:run.font.name='Courier New';run.font.size=Pt(8.5)
    def result(a,b,data=rows):return next(r for r in data if r['train_source']==a and r['test_source']==b)
    stanford=result('stanford','stanford');upenn=result('upenn','upenn')
    order=[('upenn','upenn'),('stanford','stanford'),('upenn','stanford'),('stanford','upenn')]

    page('Kết quả thực tế từ phiên Kaggle do người dùng cung cấp')
    p('Log Kaggle ghi nhận toàn bộ notebook hoàn tất, tạo biểu đồ, kiểm tra hai model offline và xuất ZIP 171,7 MB. '
      f"Model huấn luyện Stanford đạt F1 {fmt(stanford['f1'])} trên test Stanford; model huấn luyện UPenn đạt F1 {fmt(upenn['f1'])} trên test UPenn. "
      'Đây là một lần chạy mới, được ghi riêng để không thay số liệu hoặc trọng số của phiên Mac đang dùng trong ứng dụng.')
    table(document,['Train → test','Precision','Recall','F1 test','F1 0,5','Ngưỡng'],
        [[a+' → '+b,fmt(result(a,b)['precision']),fmt(result(a,b)['recall']),fmt(result(a,b)['f1']),
          fmt(result(a,b)['default_threshold_metrics']['f1']),fmt(result(a,b)['threshold'])] for a,b in order],
        [5.8,2.24,2.24,2.24,2.24,2.24])
    p('Cột F1 test dùng ngưỡng được chọn từ OOF train nguồn; không phải điểm F1 trên tập OOF. '
      'Giá trị 0,6387 là UPenn → Stanford, thuộc đánh giá chuyển nguồn. Giá trị dùng cho Stanford → Stanford là 0,7997, ở dòng FINAL TEST 1280 của log.')
    h('Đối chiếu các phương pháp trên test Stanford của cùng phiên Kaggle')
    names={'tfidf_logreg':'TF-IDF + Logistic Regression','tfidf_svm':'TF-IDF + SVM','tfidf_nb':'TF-IDF + Naive Bayes',
           'tfidf_dummy':'Dummy đa số','minilm_logreg':'MiniLM frozen + LR'}
    baseline=[]
    for name,matrix in environment['stanford_baseline_confusion'].items():
        (tn,fp),(fn,tp)=matrix
        baseline.append([names[name],fmt(tp/(tp+fp) if tp+fp else 0),fmt(tp/(tp+fn)),fmt(2*tp/(2*tp+fp+fn))])
    baseline.append(['MiniLM fine-tuned',fmt(stanford['precision']),fmt(stanford['recall']),fmt(stanford['f1'])])
    table(document,['Phương pháp','Precision','Recall','F1'],baseline,[8,3,3,3])
    (tn,fp),(fn,tp)=environment['stanford_baseline_confusion']['tfidf_logreg'];lr_f1=2*tp/(2*tp+fp+fn)
    p(f"F1 tăng {(stanford['f1']-lr_f1)*100:.2f} điểm phần trăm so với Logistic Regression trong cùng log. "
      'Các chỉ số baseline trong bảng được tính lại từ ma trận nhầm lẫn đã in trong log, không suy đoán từ số làm tròn.')
    figure(document,'kaggle_comparison.png','Hình 21. Hai lần chạy được giữ riêng; đồ thị tạo lại từ JSON của Mac và các dòng FINAL TEST Kaggle.')

    page('Ma trận và giải thích ngưỡng trong phiên Kaggle')
    figure(document,'kaggle_stanford_confusion.png','Hình 22. Stanford → Stanford trên Kaggle tại ngưỡng OOF nguồn và tại 0,5; số đếm trích từ log.')
    table(document,['Train → test','Macro F1','Accuracy','AP','ROC AUC','P@20','R@20'],
        [[a+' → '+b,*[fmt(result(a,b)[key]) for key in ['macro_f1','accuracy','average_precision','roc_auc','precision_at_20','recall_at_20']]]
         for a,b in order],[5.0,2.0,2.0,2.0,2.0,2.0,2.0])
    p('Ở Stanford → Stanford, ma trận là [[4406, 303], [167, 938]]: model tìm đúng 938 bài ưu tiên, bỏ sót 167 bài và gắn ưu tiên nhầm 303 bài. '
      'Precision = 938 / (938 + 303); Recall = 938 / (938 + 167); F1 = 1876 / (1876 + 303 + 167) = 0,799659. '
      'Đối soát từ số đếm xác nhận F1 trong log, không phụ thuộc cách định dạng phần trăm.')
    p('Ngưỡng 0,714086 của model Stanford được chọn trên OOF Stanford. Ngưỡng 0,682123 thuộc model UPenn. '
      'Không hoán đổi hai ngưỡng và không lấy số của hai model khác nhau để trình bày như cùng một phép đánh giá.')
    p('Ở UPenn → Stanford, F1 tại ngưỡng nguồn là 0,638743, thấp hơn F1 tại 0,5 là 0,665505. '
      'Ngưỡng tối ưu trên dữ liệu nguồn không bảo đảm tối ưu trên nguồn đích khi phân bố khác nhau. '
      'Báo cáo giữ cả hai kết quả; không thay ngưỡng triển khai chỉ vì đã nhìn thấy test đích.')
    p('Model Stanford trong phiên Kaggle chọn weighted_aug, trong khi phiên Mac chọn plain. '
      'Quy tắc chọn vẫn dựa trên F1 validation tại checkpoint được giữ theo AP, không dựa trên epoch có F1 cao nhất trong mọi dòng log. '
      'Vì vậy không lấy riêng F1 của epoch cuối để suy ra cấu hình phải được chọn.')
    gap=(result('stanford','stanford',local)['f1']-stanford['f1'])*100
    p(f"Stanford F1 của hai phiên chênh {gap:.2f} điểm phần trăm. Thiết bị, precision, phiên bản thư viện và cấu hình được chọn đều khác nhau. "
      'Không quy toàn bộ chênh lệch cho GPU hoặc coi hai lần chạy là một thí nghiệm đối chứng phần cứng; cũng chưa có kiểm định ý nghĩa thống kê.')

    page('Môi trường thời gian và tái tạo minh chứng Kaggle')
    table(document,['Thành phần','Phiên Kaggle theo log'],[
        ['Python',environment['python']],['PyTorch',environment['torch']],['Transformers',environment['transformers']],
        ['scikit-learn và NumPy',environment['sklearn']+' và '+environment['numpy']],
        ['GPU nhìn thấy và GPU dùng','2 Tesla T4; một GPU được dùng cho mỗi lượt fine-tuning'],
        ['Precision fine-tuning','FP32; bf16 = false'],['Cấu hình được chọn','weighted_aug cho cả UPenn và Stanford'],
        ['Thời gian toàn log',f"{environment['total_log_seconds']/60:.1f} phút đến xuất HTML"]],[7,10])
    p('Phiên Mac của phần trước dùng Python 3.12, PyTorch 2.15, Transformers 4.57 và MPS bfloat16. '
      'Cell cài thư viện của notebook kiểm tra gói có mặt, chưa ép mọi phiên bản trùng môi trường Mac; log Kaggle cho thấy Transformers 5.20 và scikit-learn 1.6. '
      'Cần lưu notebook đã chạy, metadata, split IDs và model ZIP cùng nhau khi nộp để xác định đúng phiên thực nghiệm.')
    h('Chẩn đoán thời gian và cảnh báo')
    p('Đoạn 455 batch là mã hóa embeddings Stanford frozen, kéo dài 16 phút 48 giây. '
      'Mã train.py của phiên đó chỉ chọn MPS hoặc CPU, nên phần này chạy CPU trên Kaggle dù CUDA có sẵn. '
      'Fine-tuning có bộ chọn CUDA riêng và đã dùng T4. Mã notebook hiện tại đã sửa train.py để ưu tiên CUDA; '
      'sửa này áp dụng cho lần chạy sau, không đổi kết quả log đã cung cấp.')
    p('Thông báo classifier.weight và classifier.bias MISSING khi khởi tạo mỗi lượt fit là head nhị phân mới, '
      'sau đó được học bằng nhãn urgency. Cảnh báo token vượt 256 phát sinh khi đếm chiều dài gốc để audit; '
      'đường encode thực tế cắt theo giới hạn model. Log tiếp tục hoàn tất, không có traceback hoặc lỗi indexing.')
    p('Cảnh báo debugger frozen modules liên quan breakpoint Python; không phải frozen MiniLM encoder. '
      'Thông báo không có HF_TOKEN chỉ nhắc giới hạn tải ẩn danh; model public đã tải thành công. '
      'Các SyntaxWarning cuối log thuộc bước chuyển notebook sang HTML, không phải lỗi huấn luyện.')
    h('Nguồn và lệnh kiểm tra lại')
    p('provided_log_excerpt.txt giữ bốn dòng FINAL TEST được chép từ log người dùng cung cấp; environment.json ghi metadata và ma trận baseline trong cùng log. '
      'summarize_kaggle_run.py parse JSON, tính lại P/R/F1/accuracy từ ma trận rồi tái tạo hai hình. '
      'Chưa nhập ZIP model Kaggle hoặc CSV từng dự đoán về máy; ứng dụng local vẫn dùng trọng số của phiên Mac.')
    code("bash -c 'source Project/scripts/launchers/env.sh; Project/.venv/bin/python \\\nProject/scripts/summarize_kaggle_run.py'\nbash Project/scripts/launchers/reproduce.sh")
    p('Muốn tái huấn luyện phiên Kaggle: import notebook hiện tại, bật Internet và GPU T4 ×2 rồi Run All. '
      'Các kết quả mới phải ghi thành một phiên riêng. Không cam kết số học hoặc quyết định chọn cấu hình giống hệt giữa mọi phiên bản thư viện và thiết bị.')
