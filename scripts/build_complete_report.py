"""Generate the full research report from actual saved experiments and browser evidence."""
from __future__ import annotations
import csv,hashlib,json,sys
from pathlib import Path
from docx import Document
from docx.shared import Cm,Pt
from docx.oxml.ns import qn
from build_report import style_document,table,figure,canonical_save

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';E=R/'experiments'
NAMES={'tfidf_logreg':'TF-IDF + LR','tfidf_svm':'TF-IDF + SVM','tfidf_nb':'TF-IDF + NB','minilm_logreg':'MiniLM + LR','tfidf_dummy':'Dummy đa số'}
ORDER=['tfidf_dummy','tfidf_nb','tfidf_logreg','tfidf_svm','minilm_logreg']

def load(path):return json.loads(path.read_text(encoding='utf-8'))
def f(v):return f'{v:.3f}'
def pct(v):return f'{100*v:.2f}%'
def count(v):return f'{v:,}'.replace(',','.')

def main():
    audit=load(R/'data_audit.json');results=load(E/'results.json');cv=load(E/'cv_results.json');splits=load(E/'split_info.json')
    encoder=load(E/'encoder_info.json');deploy=load(ROOT/'models/deployment.json');errors=load(E/'error_examples.json')
    browser=load(R/'screenshots/browser_checks.json');config=load(E/'config.json')
    assert len(results)==20 and len(cv)==10
    d=Document();style_document(d)
    d.sections[0].footer.paragraphs[0].runs[0].text='Đồ án ưu tiên phản hồi bài đăng  |  '
    d.core_properties.title='Ưu tiên phản hồi bài đăng sinh viên bằng học máy'
    d.core_properties.author='Báo cáo thực nghiệm đồ án học máy'
    def p(text):d.add_paragraph(text)
    def h(text):d.add_heading(text,2)
    def page(title):d.add_page_break();d.add_heading(title,1)
    def img(path,caption,width=17):
        paragraph=d.add_paragraph();paragraph.paragraph_format.keep_with_next=True
        paragraph.add_run().add_picture(str(path),width=Cm(width))
        paragraph=d.add_paragraph(caption)
        for run in paragraph.runs:run.italic=True;run.font.size=Pt(8.5)
    def code(text):
        paragraph=d.add_paragraph(text);paragraph.paragraph_format.space_after=Pt(6)
        for run in paragraph.runs:run.font.name='Courier New';run.font.size=Pt(9)
    def row(source,target,model):return next(r for r in results if r['train_source']==source and r['test_source']==target and r['model']==model)
    def metrics_table(rows,transfer=False):
        headers=['Mô hình','P','R','F1','Macro F1','AP']
        records=[[NAMES[r['model']],*[f(r[k]) for k in ['precision','recall','f1','macro_f1','average_precision']]] for r in rows]
        table(d,headers,records,[5.5,2.3,2.3,2.3,2.3,2.3])
    selected=next(c for c in cv if c['source']==deploy['source'] and c['model']==deploy['model'])
    own=row(deploy['source'],deploy['source'],deploy['model']);cross=row(deploy['source'],'upenn',deploy['model'])

    d.add_paragraph('Ưu tiên phản hồi bài đăng sinh viên bằng học máy','Title')
    d.add_paragraph('Báo cáo phương pháp thực nghiệm và sản phẩm ứng dụng','Subtitle')
    p('Đồ án xây dựng công cụ hỗ trợ giảng viên lựa chọn bài đăng cần phản hồi trước trên diễn đàn học tập. '
      'Từ nội dung tiếng Anh, hệ thống dự đoán mức ưu tiên nhị phân và xếp bài theo điểm mô hình. '
      'Báo cáo trình bày dữ liệu, phương pháp học máy đã học, thực nghiệm chuyển nguồn, phần mở rộng pretrained và minh chứng sản phẩm chạy thật.')
    h('Kết quả chính của lần chạy')
    p(f"Sau làm sạch có {count(audit['datasets']['upenn']['counts']['final_rows'])} bài UPenn và {count(audit['datasets']['stanford']['counts']['final_rows'])} bài Stanford. "
      'Trong phần thực nghiệm gốc, bốn phương pháp chính và một baseline được đánh giá theo bốn hướng, tạo 20 bộ kết quả kiểm tra độc lập với quá trình chọn mô hình trong từng nguồn.')
    table(d,['Hạng mục','Kết quả đo'],[
        ['Mô hình mặc định của ứng dụng',NAMES[deploy['model']]+' huấn luyện Stanford'],
        ['Quy tắc lựa chọn','F1 lớp ưu tiên trên dự đoán OOF của tập train Stanford'],
        ['F1 trên test Stanford',f(own['f1'])],['F1 khi chuyển sang test UPenn',f(cross['f1'])],
        ['Precision và Recall trên Stanford',f(own['precision'])+' và '+f(own['recall'])],
        ['Ngưỡng ứng dụng',f(deploy['threshold'])],['Sản phẩm','Web local tại http://127.0.0.1:8765']], [7,10])
    h('Cách đọc báo cáo')
    p('Các trang đầu giải thích yêu cầu, dữ liệu, cách chia và cơ sở thuật toán. Phần thực nghiệm đối chiếu Precision, Recall, F1, Macro-F1, AP và ma trận nhầm lẫn. '
      'Phần cuối phân tích dự đoán sai, trình bày ảnh ứng dụng và các lệnh tái tạo. Các chỉ số trong bảng được đọc từ JSON của lần huấn luyện, không nhập thủ công.')
    p('Kết quả phản ánh nhãn độ khẩn cấp của bộ dữ liệu. Mô hình chưa xác nhận bài đã được trả lời, chưa hiểu toàn bộ luồng thảo luận và chưa được kiểm chứng trên tiếng Việt.')
    if (R/'fine_tuning/results.json').exists():
        extension=load(R/'fine_tuning/results.json')
        up=next(r for r in extension if r['train_source']=='upenn' and r['test_source']=='upenn')
        st=next(r for r in extension if r['train_source']=='stanford' and r['test_source']=='stanford')
        p(f"Bổ sung fine-tuning: báo cáo hiện có tổng cộng 24 bộ kết quả test. Classifier MiniLM fine-tuned đạt F1 {f(up['f1'])} trên UPenn "
          f"và {f(st['f1'])} trên Stanford. Các chương bổ sung cuối báo cáo ghi đủ scout thất bại, 5-fold, early stopping, augmentation, model độc lập và notebook Kaggle.")

    page('Đối chiếu yêu cầu giảng viên và phạm vi đồ án')
    img(R/'reference/teacher_requirements.png','Hình 1. Ảnh yêu cầu giảng viên do sinh viên cung cấp.',16)
    table(d,['Yêu cầu trong ảnh','Phần thực hiện và bằng chứng'],[
        ['Tự tìm tối thiểu hai bộ dữ liệu','UPenn và Stanford MOOCPosts; URL, commit cố định, SHA-256, CSV gốc và CSV xử lý.'],
        ['Tổng hợp kiến thức đã học','TF-IDF kết hợp LR, SVM và NB; chia dữ liệu, cross-validation, chọn tham số và đánh giá.'],
        ['Phân tích kết quả và lỗi mô hình','Các kết quả test; P/R/F1, ma trận nhầm lẫn, PR curves, dự đoán sai có ID và nguyên văn.'],
        ['Bổ sung kiến thức ngoài môn','MiniLM pretrained; phần bổ sung thử fine-tuning, 5-fold và augmentation trên cùng test.'],
        ['Phần sản phẩm do đồ án đề xuất','Nhập câu tự do, tải CSV, xếp hàng đợi, xuất kết quả và trang số liệu.']], [6.2,10.8])
    p('Ảnh yêu cầu không quy định trước mức F1 phải đạt. Precision, Recall, F1 và ma trận nhầm lẫn là các chỉ số đồ án lựa chọn để chứng minh kết quả theo kế hoạch ban đầu. '
      'AP và Precision@20/Recall@20 được bổ sung vì mục tiêu sản phẩm là tìm các bài cần ưu tiên trong một danh sách có giới hạn thời gian đọc.')
    p('Đối chiếu tài liệu môn học: Slide3 trình bày Logistic Regression và đánh giá xếp hạng; Slide5 trình bày mô hình xác suất và Gaussian Naive Bayes; '
      'Slide7 trình bày SVM; Model-assessment trình bày đánh giá, tách model selection khỏi test và chọn ngưỡng. Multinomial NB là biến thể phù hợp đặc trưng văn bản không âm. '
      'MiniLM và sentence embeddings là phần mở rộng. Đây là ứng dụng mô hình pretrained, không phải tuyên bố mô hình mới nhất hay phương pháp mới do đồ án phát minh.')

    page('Hai nguồn dữ liệu và định nghĩa nhãn')
    table(d,['Thuộc tính','UPenn','Stanford'],[
        ['Quy mô công bố','3.503 bài từ 9 MOOC','29.604 bài từ 11 MOOC'],
        ['Dòng đọc từ tệp cố định',*[count(audit['datasets'][s]['counts']['raw_rows']) for s in ['upenn','stanford']]],
        ['Bài sau làm sạch',*[count(audit['datasets'][s]['counts']['final_rows']) for s in ['upenn','stanford']]],
        ['Bài lớp ưu tiên',*[count(audit['datasets'][s]['label_counts']['1']) for s in ['upenn','stanford']]],
        ['Trường metadata','id và post_time','CourseType; không có ID tác giả hoặc luồng trong bản CSV'],
        ['Nhãn gốc','Urgency_1_7','Urgency_1_7; có cả mức nửa điểm']], [5.5,5.75,5.75])
    p('Hai tệp được lấy từ kho bổ trợ nghiên cứu của Švábenský và cộng sự. Dùng cùng một kho tải không làm hai tập thành cùng nguồn: '
      'bài đăng gốc đến từ hai trường và những khóa học khác nhau. Nguồn Stanford được ghi nhận cho Agrawal và Paepcke. '
      'Kho có mã và dữ liệu tham khảo; không suy diễn giấy phép MIT của mã thành quyền phát hành lại mọi dữ liệu.')
    p('Quy đổi cố định trước thực nghiệm: priority = 1 khi urgency > 4, ngược lại bằng 0. Quy tắc này theo mã bổ trợ nghiên cứu. '
      'Ví dụ 4 là lớp 0, còn 4,5 là lớp 1. Không làm tròn nhãn Stanford trước khi quy đổi. Độ khẩn cấp là đại diện cho nhu cầu ưu tiên, không phải xác nhận bắt buộc phải phản hồi.')
    figure(d,'02_labels.png','Hình 2. Phân phối nhãn sau xử lý; tạo lại từ reports/data_audit.json.')
    p('Tỷ lệ dương của UPenn thấp hơn Stanford. Vì vậy accuracy cao vẫn có thể đi kèm việc bỏ sót hầu hết bài cần ưu tiên. '
      'Đồ án giữ baseline dự đoán đa số để minh họa vấn đề và báo cáo riêng F1 lớp dương cùng Macro-F1.')

    page('Xử lý dữ liệu và kiểm soát chất lượng')
    p('scripts/prepare_data.py tải tệp theo commit cố định, kiểm tra SHA-256 và schema, rồi giữ cả text_raw lẫn text_clean. '
      'Văn bản được chuẩn hóa NFC, giải mã thực thể HTML, bỏ một số thẻ định dạng và gộp khoảng trắng. Dấu câu và từ phủ định được giữ vì có thể mang thông tin về khó khăn học tập.')
    reasons=[('Dòng thiếu trường','malformed_or_missing_fields'),('Bản trùng cùng nhãn trong nguồn','duplicate_within_source'),('Dòng thuộc nhóm nhãn mâu thuẫn','conflicting_binary_labels'),('Bản trùng giữa hai nguồn','cross_source_duplicates')]
    table(d,['Quyết định lọc','UPenn','Stanford'],[[name,*[audit['datasets'][s]['counts'].get(key,0) for s in ['upenn','stanford']]] for name,key in reasons],[9,4,4])
    p('Trùng được so theo văn bản đã casefold và gộp khoảng trắng. Nhóm trùng cùng nhãn giữ bản đầu; nhóm có nhãn nhị phân mâu thuẫn bị loại toàn bộ. '
      'Bản trùng giữa nguồn được giữ tại UPenn và loại khỏi Stanford theo quy tắc cố định. Đây là bước xây dựng tập đã làm sạch trước chia; '
      'kết luận đánh giá áp dụng cho tập sau lọc, không trực tiếp cho toàn bộ phân phối dữ liệu gốc.')
    figure(d,'03_lengths.png','Hình 3. Độ dài đo bằng số từ, chưa phải số token MiniLM.')
    p('Mỗi dòng loại được ghi nguồn, số thứ tự bản ghi và lý do trong rejected_rows.csv. Các assert kiểm tra đối soát tổng dòng, nhãn hợp lệ, văn bản không rỗng, '
      'đủ hai lớp và không còn trùng chính xác trong hay giữa nguồn. So trùng này chưa phát hiện diễn đạt gần nghĩa, trích dẫn hoặc các bài cùng chủ đề.')
    p('Nguồn UPenn có một dòng thiếu trường ngoài 3.503 bản ghi hợp lệ. Vì vậy số dòng đọc là 3.504 trước khi lọc. '
      'Việc lưu số đếm thực tế và tệp gốc giúp giải thích chênh lệch với con số công bố.')

    page('Chia tập và thiết kế đánh giá không dùng test để chọn mô hình')
    table(d,['Nguồn','Train','Test','Dương train','Dương test'],[[s,*[splits[s][k] for k in ['train','test','train_positives','test_positives']]] for s in ['upenn','stanford']],[4,3.25,3.25,3.25,3.25])
    p('Seed được cố định bằng 42. UPenn chia theo ID sinh viên bằng GroupShuffleSplit với tỷ lệ nhóm test 20%; tỷ lệ số bài thực tế có thể khác 20%. '
      'Không sinh viên nào xuất hiện đồng thời ở train và test. Trong train, StratifiedGroupKFold ba fold cố gắng bảo toàn tỷ lệ lớp và giữ sinh viên trong một fold.')
    p('Bản Stanford đang dùng không có ID tác giả hay luồng. Do đó dùng hold-out 20% phân tầng theo nhãn, và StratifiedKFold ba fold trong train. '
      'Giới hạn là chưa ngăn được các bài liên quan của cùng tác giả/luồng giữa hai phần. CourseType chỉ là miền chủ đề rộng, không được coi là ID khóa học.')
    table(d,['Nguồn huấn luyện','Nguồn kiểm tra','Ý nghĩa'],[['UPenn','UPenn','Khả năng trên sinh viên UPenn chưa gặp'],['Stanford','Stanford','Khả năng trên bài Stanford giữ lại'],['UPenn','Stanford','Chuyển sang nguồn khác không tinh chỉnh đích'],['Stanford','UPenn','Chuyển nguồn theo chiều ngược lại']],[5,5,7])
    h('Thứ tự thực hiện')
    p('1. Cố định tập test và lưu post_id trong split_ids.csv. 2. Tìm tham số bằng CV trên train nguồn. '
      '3. Với cấu hình tốt nhất, sinh điểm out-of-fold trên train nguồn để chọn ngưỡng. 4. Fit cấu hình đó trên toàn bộ train nguồn. '
      '5. Dùng cùng mô hình và ngưỡng để đánh giá test của cả hai nguồn. Không huấn luyện lại TF-IDF trên test đích.')
    p('Pipeline đặt TF-IDF bên trong từng fold CV: từ vựng và IDF chỉ được học trên phần train của fold. '
      'MiniLM giữ nguyên trọng số, không học từ nhãn của tập test. Dự đoán OOF phục vụ lựa chọn ngưỡng; số liệu OOF đó không được trình bày như một tập test độc lập.')

    page('Biểu diễn TF IDF và các thuật toán trong môn học')
    h('TF IDF')
    p('Mỗi văn bản được biểu diễn bằng unigram và bigram. TF đo mức xuất hiện của từ trong bài; IDF giảm trọng số từ phổ biến toàn tập train. '
      'Với sublinear_tf, tf = 1 + log(count). IDF có làm trơn: idf(t) = log((1 + N)/(1 + df(t))) + 1. Vector cuối được chuẩn hóa L2. '
      'Tham số cố định: min_df = 2, max_df = 0,98, tối đa 40.000 đặc trưng, chữ thường và strip_accents=unicode; không bỏ stopwords để giữ từ phủ định.')
    h('Logistic Regression')
    p('Mô hình tính z = wᵀx + b rồi p = 1/(1 + exp(-z)). Tối ưu log-loss có regularization. Tham số C điều khiển nghịch đảo độ mạnh regularization. '
      'Dùng class_weight=balanced để tăng trọng số lớp hiếm; liblinear, max_iter=2000 và seed 42. Điểm p được dùng xếp hạng và so với ngưỡng đã chọn; '
      'chưa có nghiên cứu hiệu chuẩn để diễn giải p như xác suất tuyệt đối đáng tin cậy trong mọi nguồn mới.')
    h('SVM tuyến tính')
    p('LinearSVC học siêu phẳng phân tách với biên và squared hinge loss mặc định. Điểm wᵀx + b là khoảng quyết định, không phải xác suất. '
      'Dùng class_weight=balanced, max_iter=10000. Thử C = 0,3; 1; 3. Khi xếp hàng đợi, điểm cao hơn thể hiện bằng chứng mạnh hơn cho lớp ưu tiên trong cùng mô hình.')
    h('Multinomial Naive Bayes')
    p('NB so sánh log P(c) + tổng xⱼ log P(tⱼ|c), giả định đặc trưng độc lập có điều kiện theo lớp. Dùng TF-IDF không âm như một lựa chọn thực nghiệm cho văn bản. '
      'Thử alpha = 0,1; 0,5; 1 để làm trơn. Đây là Multinomial NB, khác Gaussian NB dùng phân phối Gaussian trong slide. Không truyền đặc trưng âm của embeddings vào Multinomial NB.')
    h('Baseline và so sánh công bằng')
    p('DummyClassifier(strategy=prior) dự đoán theo phân phối train và cho cùng một điểm cho mọi bài; ở ngưỡng mặc định, nó dự đoán lớp đa số. '
      'Baseline giúp xác định mô hình có học tín hiệu hữu ích hay chỉ tận dụng mất cân bằng lớp. Tất cả mô hình dùng cùng train/test và cùng quy tắc chọn ngưỡng, '
      'ngoại trừ Dummy giữ ngưỡng 0,5 để thể hiện hành vi đa số.')

    page('Phần mở rộng dùng MiniLM pretrained')
    p('Sentence Transformer all-MiniLM-L6-v2 chuyển câu hoặc đoạn văn tiếng Anh thành vector 384 chiều. '
      'Đồ án tải mô hình pretrained tại revision cố định, giữ nguyên trọng số encoder và chuẩn hóa embeddings. Sau đó huấn luyện Logistic Regression trên nhãn của UPenn hoặc Stanford. '
      'Chỉ bộ phân loại được fit bằng dữ liệu đồ án; không pretrain hay fine-tune toàn bộ Transformer.')
    code('văn bản → tokenizer MiniLM → encoder cố định → vector 384 chiều\n→ Logistic Regression → điểm → ngưỡng → nhãn ưu tiên')
    table(d,['Thành phần','Cấu hình thực tế'],[['Mô hình',encoder['model']],['Revision',encoder['revision']],['Thiết bị tạo embeddings',encoder['device']],['Số chiều',encoder['dimension']],['Giới hạn token',encoder['max_seq_length']],['Regularization C của LR','0,3; 1; 3; 10']],[5,12])
    p('Giới hạn chiều dài được đo bằng tokenizer thật trước truncation, có tính special tokens. Thống kê bên dưới khác số từ tách khoảng trắng ở phần EDA. '
      'Trong thí nghiệm này, phần vượt max_seq_length bị cắt theo cấu hình mặc định của encoder; không tuyên bố mô hình đã đọc toàn bộ mọi bài.')
    table(d,['Nguồn','Bài bị cắt','Tổng bài','Tỷ lệ','P95 token'],[[s,encoder['sources'][s]['truncated_posts'],encoder['sources'][s]['total'],pct(encoder['sources'][s]['truncated_posts']/encoder['sources'][s]['total']),f(encoder['sources'][s]['tokens_p95'])] for s in ['upenn','stanford']],[4,3.25,3.25,3.25,3.25])
    p('Embeddings được cache theo hash CSV xử lý và revision mô hình. Chạy lại có thể dùng cache để tránh mã hóa lại hàng chục nghìn bài. '
      'Vì encoder cố định và không học nhãn, việc mã hóa từng văn bản trước CV không tạo ra bước fit giám sát trên test. Bộ LR vẫn được tìm tham số và ngưỡng chỉ từ train nguồn.')
    p('Đây là kiến thức ngoài phần thuật toán truyền thống trong bộ slide đã cung cấp. Giá trị của phần mở rộng được đánh giá bằng kết quả trên cùng tập test và khả năng chuyển nguồn, '
      'không mặc định rằng pretrained luôn tốt hơn TF-IDF. MiniLM là mô hình nền tham khảo, không phải mô hình mới nhất tại thời điểm nộp.')

    page('Chọn siêu tham số và ngưỡng quyết định')
    p('GridSearchCV dùng ba fold và scoring=f1 cho lớp dương với ngưỡng mặc định: LR/NB 0,5; SVM 0. Cấu hình có F1 CV trung bình cao nhất được fit lại trên train nguồn. '
      'Sau đó cross_val_predict tạo điểm OOF của cấu hình đã chọn để tìm ngưỡng tối đa hóa F1 lớp ưu tiên. Nếu F1 bằng nhau, chọn ngưỡng lớn nhất.')
    table(d,['Nguồn và mô hình','Tham số tốt','F1 CV','F1 OOF sau ngưỡng','Ngưỡng'],[[c['source']+' '+NAMES[c['model']],str(c['best_params']).replace('classifier__',''),f(c['cv_mean_f1']),f(c['oof_tuned_f1']),f(c['threshold'])] for c in cv if c['name']!='dummy'],[6,3.4,2.3,3,2.3])
    figure(d,'threshold.png','Hình 4. Đường Precision, Recall, F1 theo ngưỡng trên OOF train của mô hình triển khai. Test không được dùng vẽ đường để chọn ngưỡng.')
    p('Các giá trị OOF đã tham gia lựa chọn cấu hình và ngưỡng nên có thể lạc quan. Tập test giữ lại mới dùng đánh giá sau lựa chọn. '
      'Mô hình web được chọn trong nhóm huấn luyện Stanford dựa trên F1 OOF vì đây là nguồn lớn hơn; không chọn dựa trên mô hình có F1 test đẹp nhất. '
      'Chọn Stanford là quyết định thiết kế đã ghi trong config.json, không chứng minh đây là nguồn tốt nhất cho mọi diễn đàn.')

    for source,title,number in [('upenn','Kết quả kiểm tra trong nguồn UPenn',5),('stanford','Kết quả kiểm tra trong nguồn Stanford',6)]:
        page(title)
        s=splits[source];p(f"Tập test có {s['test']} bài, trong đó {s['test_positives']} bài ưu tiên. "
            'Bảng dùng cùng nhãn thật và cùng danh sách post_id cho tất cả phương pháp. Ngưỡng của từng phương pháp được chọn trên OOF train nguồn.')
        internal=[row(source,source,m) for m in ORDER];metrics_table(internal)
        figure(d,f'{source}_metrics.png',f'Hình {number}. Precision, Recall và F1 lớp ưu tiên trên test {source}.')
        best=max(internal,key=lambda r:r['f1']);dummy=row(source,source,'tfidf_dummy')
        h('Diễn giải kết quả')
        p(f"F1 lớp ưu tiên cao nhất trong bảng test là {NAMES[best['model']]} với {f(best['f1'])}. Đây là mô tả kết quả hậu nghiệm trên test, "
          'không thay đổi quy tắc lựa chọn mô hình triển khai. '
          f"Baseline đa số có accuracy {pct(dummy['accuracy'])}, nhưng Recall {f(dummy['recall'])} và F1 {f(dummy['f1'])}. "
          'Điều đó cho thấy accuracy đơn lẻ không đủ để đánh giá công cụ tìm bài cần phản hồi.')
        lr=row(source,source,'tfidf_logreg');svm=row(source,source,'tfidf_svm');nb=row(source,source,'tfidf_nb')
        p(f"LR đạt Precision/Recall {f(lr['precision'])}/{f(lr['recall'])}; SVM đạt {f(svm['precision'])}/{f(svm['recall'])}; "
          f"NB đạt {f(nb['precision'])}/{f(nb['recall'])}. Precision thấp tạo thêm bài không khẩn cấp trong hàng đợi; Recall thấp khiến bài cần hỗ trợ bị bỏ sót. "
          'Lựa chọn sử dụng thực tế phải cân nhắc thời gian giảng viên và chi phí bỏ sót, thay vì chỉ nhìn một cột điểm.')

    page('Kết quả chuyển mô hình giữa hai nguồn dữ liệu')
    p('Ở mỗi chiều, mô hình và ngưỡng nguồn được giữ nguyên. Chỉ dùng phần test cố định của nguồn đích; không điều chỉnh theo nhãn đích. '
      'Hai chiều có kích thước train và tỷ lệ lớp khác nhau nên không diễn giải như hai thí nghiệm đối xứng hoàn toàn.')
    table(d,['Chiều và phương pháp','P','R','F1','Macro F1'],[[a+' → '+b+' '+NAMES[m],*[f(row(a,b,m)[k]) for k in ['precision','recall','f1','macro_f1']]] for a,b in [('upenn','stanford'),('stanford','upenn')] for m in ORDER[1:]],[7,2.5,2.5,2.5,2.5])
    figure(d,'transfer.png','Hình 7. F1 lớp ưu tiên ở hai chiều chuyển nguồn; điểm lấy từ test chưa dùng để tuning.')
    p('Chuyển nguồn làm thay đổi chủ đề khóa học, cách viết và tỷ lệ khẩn cấp. TF-IDF có thể phụ thuộc vào từ khóa đặc thù nguồn; MiniLM có thể chia sẻ biểu diễn ngữ nghĩa, '
      'nhưng vẫn bị giới hạn bởi nhãn nguồn, nội dung bị cắt và kiểu diễn đạt chưa gặp. Đây là các giả thuyết giải thích cần được đối chiếu với lỗi cụ thể, không phải quan hệ nhân quả đã chứng minh.')
    p('So sánh trong nguồn và chuyển nguồn còn chịu ảnh hưởng của độ khó tập test đích. Vì vậy giảm hoặc tăng F1 giữa hai tập khác nhau không đo riêng một yếu tố domain shift. '
      'Để kết luận chắc hơn cần thêm nguồn, nhiều seed và đánh giá leave-one-course-out khi có đủ metadata.')

    page('Ma trận nhầm lẫn và cách tính các chỉ số')
    figure(d,'confusion_four.png','Hình 8. Bốn ma trận chọn theo OOF trong phần thực nghiệm gốc. Hàng là nhãn thật; cột là dự đoán. Không chọn mô hình bằng test.')
    tn,fp=own['confusion_matrix'][0];fn,tp=own['confusion_matrix'][1]
    p(f"Ví dụ kiểm chứng cho mô hình web trên Stanford: TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}. "
      f"Tổng {tn+fp+fn+tp} khớp kích thước tập test. FP là bài bị đưa lên ưu tiên nhầm; FN là bài khẩn cấp bị bỏ sót.")
    table(d,['Chỉ số','Thay số từ ma trận','Kết quả'],[
        ['Precision','TP / (TP + FP) = '+f'{tp}/({tp}+{fp})',f(own['precision'])],
        ['Recall','TP / (TP + FN) = '+f'{tp}/({tp}+{fn})',f(own['recall'])],
        ['F1','2TP / (2TP + FP + FN) = '+f'{2*tp}/({2*tp}+{fp}+{fn})',f(own['f1'])],
        ['Accuracy','(TP + TN) / N',f(own['accuracy'])]], [4,9,4])
    p('Macro-F1 là trung bình F1 của hai lớp, cho mỗi lớp trọng số bằng nhau. F1 lớp ưu tiên phản ánh trực tiếp việc phát hiện bài khẩn cấp; '
      'không thay thế hai giá trị Precision và Recall khi đánh giá khả năng sử dụng của hàng đợi.')

    page('Đánh giá thứ hạng và đường Precision Recall')
    figure(d,'upenn_pr.png','Hình 9. Precision–Recall trên test UPenn. AP được tính từ điểm liên tục, không dùng nhãn dự đoán nhị phân.')
    figure(d,'stanford_pr.png','Hình 10. Precision–Recall trên test Stanford. Đường đứt thể hiện tỷ lệ lớp dương của test.')
    table(d,['Nguồn test của mô hình web','AP','P@20','R@20','Bài dương trong top 20'],[[r['test_source'],f(r['average_precision']),f(r['precision_at_20']),f(r['recall_at_20']),r['top20_found']] for r in [own,cross]],[6,2.5,2.5,2.5,3.5])
    p('P@20 = số bài dương trong 20 điểm cao nhất / 20; R@20 chia cùng số bài tìm được cho tổng bài dương của test. '
      'AP tóm tắt Precision theo các mức Recall trên toàn thứ hạng. Không tính diện tích ROC từ nhãn dự đoán thay cho điểm. '
      'Top 20 là giới hạn đọc giả định để đánh giá, không phải giới hạn của giao diện.')

    page('So sánh TF IDF với biểu diễn ngữ nghĩa pretrained')
    table(d,['Chiều đánh giá','F1 TF-IDF LR','F1 MiniLM LR','Chênh lệch'],[[a+' → '+b,f(row(a,b,'tfidf_logreg')['f1']),f(row(a,b,'minilm_logreg')['f1']),f(row(a,b,'minilm_logreg')['f1']-row(a,b,'tfidf_logreg')['f1'])] for a,b in [('upenn','upenn'),('stanford','stanford'),('upenn','stanford'),('stanford','upenn')]],[6.5,3.5,3.5,3.5])
    p('So sánh giữ cùng họ bộ phân loại LR để tập trung vào biểu diễn văn bản. Tuy nhiên LR dùng lưới C khác nhau ở hai biểu diễn và ngưỡng được tối ưu riêng; '
      'do đó đây là so sánh hai pipeline hoàn chỉnh, chưa phải thí nghiệm cô lập tuyệt đối tác động của embeddings.')
    table(d,['Nguồn và phương pháp','Thời gian tuning và fit giây'],[[c['source']+' '+NAMES[c['model']],f(c['training_seconds'])] for c in cv if c['name']!='dummy'],[11,6])
    p('Thời gian đo gồm GridSearchCV, fit cuối và sinh dự đoán OOF; không bao gồm mã hóa MiniLM hay tải model. '
      f"Mã hóa ban đầu đo riêng: UPenn {f(encoder['sources']['upenn']['encoding_seconds'])} giây và Stanford {f(encoder['sources']['stanford']['encoding_seconds'])} giây, "
      f"thiết bị {encoder['device']}. Chạy lại từ embeddings cache sẽ không tái trả chi phí này.")
    p('Điểm tốt hơn trên một lần chia chưa chứng minh ưu thế có ý nghĩa thống kê. Báo cáo chưa thực hiện kiểm định thống kê hoặc chạy nhiều seed. '
      'Các yếu tố cần phân tích thêm gồm tỷ lệ lớp, sự phù hợp ngữ nghĩa với nhiệm vụ khẩn cấp và các bài vượt giới hạn token. Không kết luận tăng tốc GPU từ số đo chưa so trực tiếp CPU/GPU.')

    page('Phân tích dự đoán sai và nguyên nhân có thể')
    p('Các ví dụ dưới lấy trực tiếp từ CSV dự đoán của mô hình web. Chọn theo quy tắc cố định: FP có điểm cao nhất, FN có điểm thấp nhất. '
      'Đây là các lỗi tự tin cao, không phải mẫu đại diện ngẫu nhiên cho mọi lỗi. Văn bản dài được trích đoạn tối đa 280 ký tự; ID dùng để tra toàn văn.')
    for target in ['stanford','upenn']:
        for kind in ['FN','FP']:
            chosen=errors[target][kind]
            if not chosen:continue
            r=chosen[0]
            h(f"{target} {kind} {r['post_id']}")
            p(f"Nhãn thật {r['label']}; dự đoán {r['predicted']}; urgency {r['urgency']}; điểm {float(r['score']):.4f}; ngưỡng {deploy['threshold']:.4f}.")
            p('“'+r['text'][:280]+('…' if len(r['text'])>280 else '')+'”')
    p('FN cho thấy điểm thấp chưa bảo đảm nội dung không cần hỗ trợ. FP cho thấy từ ngữ liên quan khó khăn có thể xuất hiện trong bài không được gán khẩn cấp. '
      'Sự khác biệt giữa urgency và “có câu hỏi”, sự phụ thuộc bối cảnh khóa học, nội dung dài hoặc nhãn chủ quan đều là các khả năng cần xem xét khi đọc toàn văn. '
      'Không tự sửa nhãn test để làm điểm cao hơn.')
    notes=E/'error_analysis_notes.json'
    if notes.exists():
        for note in load(notes):p(note)

    page('Sản phẩm web nhập câu và nhận phân loại')
    img(R/'screenshots/02_urgent.png','Hình 11. Ảnh Chrome chạy thật sau khi gửi một câu mới. Nhãn và điểm hiển thị được kiểm tra khớp JSON trả về từ API.',17)
    p('Người dùng nhập nội dung tiếng Anh, chọn mô hình và bấm Phân loại bài đăng. Backend dùng đúng phép làm sạch đã dùng khi chuẩn bị dữ liệu, '
      'tải mô hình joblib, tạo đặc trưng và so điểm với ngưỡng đã lưu. Các nút ví dụ chỉ điền nội dung để gửi qua cùng endpoint; không chứa quy tắc ép nhãn.')
    urgent=next(c for c in browser if c['case']=='urgent')
    p(f"Ở ảnh này, API trả nhãn {urgent['label']}, điểm {urgent['score']:.4f}, ngưỡng {urgent['threshold']:.4f}. "
      'Đây là minh chứng chức năng với câu do người dùng nhập, không phải một điểm đánh giá accuracy vì câu minh họa không có nhãn kiểm định độc lập.')
    p('Với mô hình tuyến tính TF-IDF, ứng dụng còn hiển thị các đặc trưng có đóng góp lớn theo trọng số. Các đóng góp này giải thích phép tính tuyến tính trong mô hình, '
      'không chứng minh nguyên nhân xã hội hay sư phạm. Với MiniLM, giao diện không bịa ra danh sách từ giải thích cho toàn bộ encoder.')

    page('Kiểm tra một nội dung khác và phạm vi dự đoán')
    img(R/'screenshots/03_thanks.png','Hình 12. Một câu khác được gửi đến mô hình; ảnh và kết quả lưu trong browser_checks.json.',17)
    thanks=next(c for c in browser if c['case']=='thanks')
    p(f"Câu minh họa thứ hai trả nhãn {thanks['label']}, điểm {thanks['score']:.4f}, ngưỡng {thanks['threshold']:.4f}. "
      'Ứng dụng cho phép nhập câu tự do thay vì chỉ phân loại danh sách có sẵn. Nhãn được tính từ mô hình, không từ việc chọn nút ví dụ nào.')
    p('Điểm LR/NB là ước lượng của bộ phân loại; điểm SVM là decision margin. Không so trực tiếp độ lớn điểm giữa hai mô hình để kết luận mức tin cậy. '
      'Ngưỡng tương ứng được hiển thị cùng kết quả. Trường hợp rỗng hoặc không có chữ cái được từ chối; mỗi bài tối đa 12.000 ký tự để giữ tác vụ demo ổn định.')
    p('Ứng dụng được huấn luyện và đo trên tiếng Anh. Câu tiếng Việt, câu ngoài ngữ cảnh học tập hoặc một câu quá ngắn có thể cho kết quả không đáng tin cậy. '
      'Cần dữ liệu có nhãn phù hợp và đánh giá riêng trước khi sử dụng cho diễn đàn tiếng Việt. Đây là hạn chế phạm vi dữ liệu, không thể giải quyết chỉ bằng đổi ngôn ngữ giao diện.')

    page('Sản phẩm xếp hàng đợi và xuất kết quả')
    img(R/'screenshots/04_queue.png','Hình 13. Tám bài mẫu được phân loại rồi xếp giảm dần theo điểm mô hình trong cùng một lần chạy.',17)
    p('Hàng đợi nhận CSV với cột text, post_text hoặc text_clean. Mỗi lần xử lý tối đa 500 bài. Backend phân loại theo mô hình đã chọn; frontend sắp điểm giảm dần. '
      'Nếu bằng điểm, giữ thứ tự đầu vào. Nút Xuất kết quả lưu CSV gồm text, priority, score và threshold để giảng viên tiếp tục xử lý.')
    p('Kiểm tra trình duyệt đã xác nhận có đủ tám dòng, thứ tự điểm không tăng và tệp xuất tải thành công. Tệp minh chứng exported_queue.csv nằm cùng ảnh. '
      'Danh sách mẫu chỉ phục vụ demo; không đưa vào tập test, không dùng tính Precision hoặc F1 trong báo cáo.')

    page('Sản phẩm hiển thị kết quả thực nghiệm')
    img(R/'screenshots/05_metrics.png','Hình 14. Trang so sánh đọc trực tiếp results.json của lần huấn luyện, kèm ma trận nhầm lẫn.',17)
    p('Người dùng chọn một trong bốn hướng đánh giá. Trang hiển thị Precision, Recall, F1 lớp ưu tiên, Macro-F1, AP và P@20, '
      'sắp theo F1 test để thuận tiện xem kết quả. Hàng cao nhất là mô tả test hậu nghiệm; quy tắc chọn mô hình mặc định của ứng dụng vẫn là OOF Stanford.')
    p('Ma trận trên trang minh họa phương pháp đứng đầu bảng của hướng đang xem. Trong phần thực nghiệm gốc, các ma trận tại Hình 8 dùng mô hình tốt theo OOF riêng từng nguồn; '
      'hai tiêu chí chọn có thể dẫn đến hai mô hình khác nhau. Sự phân biệt này tránh nhầm số liệu giữa giao diện khám phá test và quyết định triển khai.')
    p('Giao diện có bố cục responsive và đã kiểm tra ở chiều rộng 390 pixel bằng Chrome headless. Ảnh mobile được lưu trong reports/screenshots/06_mobile.png. '
      'Đây là kiểm tra hiển thị trên viewport mô phỏng, chưa phải kiểm thử thiết bị Android thật hay bản APK.')

    page('Tái tạo thí nghiệm và quản lý môi trường')
    p('Toàn bộ mã, CSV, model, embeddings cache, thư viện dự án và báo cáo nằm trong Project. Không cài bằng Homebrew, không sửa shell profile và không cài gói ML vào Python global. '
      'setup.sh tạo .venv bằng Python 3.12 đã có, cài dependencies trong thư mục đó và lưu requirements-ml.lock.txt. Runtime tạo Word có sẵn của Codex được dùng ở chế độ chỉ đọc.')
    h('Lệnh chạy sản phẩm')
    code('bash Project/scripts/launchers/start_app.sh\n# Mở http://127.0.0.1:8765 trong trình duyệt')
    h('Lệnh chạy lại quy trình đầy đủ từ machine')
    code('bash Project/scripts/launchers/reproduce.sh --train')
    p('Lệnh này chuẩn bị dữ liệu, chạy lại huấn luyện, vẽ biểu đồ, bật server tạm để kiểm tra và chụp Chrome, rồi tạo Word và manifest. '
      'Lần đầu cần Internet cho nguồn CSV và pretrained; khi mọi tệp đã có, các bước tính toán dùng dữ liệu và model local. Embeddings cache được đối soát hash nguồn và revision.')
    h('Tạo lại báo cáo từ kết quả đã lưu')
    code('bash Project/scripts/launchers/reproduce.sh\npython3 Project/scripts/verify_evidence.py')
    p('Không có --train thì giữ kết quả thực nghiệm đã lưu, tái vẽ biểu đồ và chụp lại ứng dụng. Muốn đánh giá khả năng tái chạy huấn luyện, dùng --train hoặc script kiểm tra ML. '
      'Thời gian đo và metadata môi trường có thể thay đổi; không hứa mọi byte của toàn báo cáo thực nghiệm luôn giống hệt khi khác máy, thư viện hay backend GPU.')
    repro=E/'reproducibility_ml.json'
    if repro.exists():
        check=load(repro)
        p(f"Kiểm tra chạy lại đã thực hiện: {check['experiments']} bộ kết quả giữ nguyên ma trận nhầm lẫn; chênh lệch lớn nhất của các chỉ số là {check['maximum_metric_difference']:.2g}, "
          f"với sai số cho phép {check['tolerance']}. Kiểm tra tái dùng embeddings cache, không đánh giá độ lặp lại của việc mã hóa GPU từ đầu.")
    h('Những gì được ghi lại')
    p('config.json lưu seed, quy tắc và tham số; split_ids.csv lưu tập chia; cv_results.json lưu kết quả CV; results.json lưu test; predictions_*.csv lưu từng nhãn và điểm; '
      'encoder_info.json lưu revision, token và thiết bị. evidence_manifest.json ghi SHA-256 của mã, dữ liệu, mô hình, hình, ảnh và Word để phát hiện sửa đổi sau khi chạy.')
    p('Dữ liệu đang được mở trên Word hoặc Excel có thể tạo tệp khóa riêng của ứng dụng. Không chỉnh trực tiếp báo cáo sinh tự động nếu muốn tái tạo; chỉnh mã hoặc dữ liệu nguồn được kiểm soát rồi chạy lại. '
      'Muốn gỡ dự án, dừng server và xóa Project. Các slide PDF ở thư mục cha được giữ nguyên.')

    page('Kết luận giới hạn và tài liệu tham khảo')
    p(f"Phần thực nghiệm gốc đã hoàn thành hai nguồn dữ liệu, ba mô hình học máy truyền thống, một pipeline pretrained, 20 bộ kết quả test và ứng dụng phân loại câu mới. "
      f"Mô hình mặc định {NAMES[deploy['model']]} huấn luyện Stanford đạt F1 {f(own['f1'])} trên test cùng nguồn và {f(cross['f1'])} trên test UPenn. "
      'Các giá trị này mô tả lần chia và cách gán nhãn hiện tại; không phải cam kết chất lượng cho mọi lớp học.')
    h('Giới hạn cần trình bày khi bảo vệ')
    p('Nhãn là độ khẩn cấp cảm nhận, có tính chủ quan; mô hình không biết bài đã được giải quyết. Chỉ có một hold-out seed và hai nguồn tiếng Anh. '
      'Không có ID tác giả/luồng của Stanford để chia nhóm chặt hơn. Nội dung dài bị cắt khi dùng MiniLM; chưa kiểm tra trùng gần nghĩa. '
      'Chưa đo lợi ích bằng nghiên cứu người dùng với giảng viên hoặc kiểm định thống kê cho chênh lệch mô hình.')
    h('Hướng cải tiến có căn cứ từ thí nghiệm')
    p('Ưu tiên thu thập bài có nhãn tại môi trường đích và đánh giá thêm nhiều seed; bổ sung bối cảnh luồng và trạng thái đã trả lời nếu có tại thời điểm dự đoán. '
      'Đối với bài dài, so sánh chia đoạn và pooling với truncation. Với điểm xác suất, đánh giá calibration trước khi dùng như xác suất rủi ro. '
      'Nếu muốn hỗ trợ tiếng Việt, cần một quy trình dữ liệu và test tiếng Việt; không chỉ dùng bản dịch vài ví dụ để tuyên bố hỗ trợ đầy đủ.')
    h('Tài liệu tham khảo')
    refs=[
        '[1] Švábenský và cộng sự (2023). Towards Generalizable Detection of Urgency of Discussion Forum Posts. https://arxiv.org/abs/2307.07614',
        '[2] Kho bổ trợ nghiên cứu và dữ liệu. https://github.com/pcla-code/forum-posts-urgency. Commit cố định: '+audit['source_revision'],
        '[3] Agrawal và Paepcke (2014). The Stanford MOOC Posts Dataset. Thông tin nguồn được dẫn trong README của [2].',
        '[4] Sentence Transformers. all-MiniLM-L6-v2 model card. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2',
        '[5] scikit-learn. Common pitfalls and recommended practices; Model evaluation; StratifiedGroupKFold. https://scikit-learn.org/stable/',
        '[6] Tài liệu môn học được cung cấp trong machine: Slide1, Slide2, Slide3, Slide4-KNN, Slide5-Prob-model, Slide6-RandomForest, Slide7-SVM và Model-assessment.']
    for ref in refs:
        paragraph=d.add_paragraph(ref)
        for run in paragraph.runs:run.font.size=Pt(8.5)
    for paragraph in d.paragraphs:
        for border in list(paragraph._p.findall('.//'+qn('w:pBdr'))):border.getparent().remove(border)
    from append_finetuning_report import append_fine_tuning
    append_fine_tuning(d)
    from append_kaggle_report import append_kaggle_run
    append_kaggle_run(d)
    from append_blend_report import append_blend
    append_blend(d)
    from append_reliability_report import append_reliability
    append_reliability(d)
    from append_model_comparison_report import append_model_comparison
    append_model_comparison(d)
    from append_final_minilm_report import append_final_minilm
    append_final_minilm(d)
    target=R/'bao_cao_do_an_hoc_may.docx';canonical_save(d,target)
    print('Created full report:',target)

if __name__=='__main__':
    from build_official_report import main as build_official
    build_official()
