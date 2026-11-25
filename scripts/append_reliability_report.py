"""Append measured joint-model reliability and the three-dataset workflow to the same DOCX."""
import csv,json
from pathlib import Path
from docx.shared import Cm,Pt
from build_report import table,figure
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';OUT=R/'reliability/deberta_real_only'
def load(path):return json.loads(path.read_text())
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def append_reliability(d):
    if not (OUT/'results.json').exists():return
    protocol=load(OUT/'protocol.json');plan=load(R/'reliability/data_plan.json');audit=load(OUT/'reliability_audit.json')
    results=load(OUT/'results.json');mixed={r['test_source']:r for r in results if r['policy']=='mixed'}
    custom=load(ROOT/'Datasets/CustomForum/metadata.json');config=protocol['configuration']
    def p(text):d.add_paragraph(text)
    def page(title):d.add_page_break();d.add_heading(title,1)
    def f(value):return f'{value:.4f}'
    def percent(value):return f'{value*100:.1f}%'
    page('Mục tiêu phân loại đáng tin cậy và ba bộ dữ liệu')
    p('Mục tiêu của phần mở rộng là giúp giảng viên nhận diện đúng nhu cầu hỗ trợ hiện tại, đồng thời quan sát những trường hợp mô hình còn sai. '
      'Chúng tôi thử DeBERTa v3 base với ngữ cảnh dài hơn và thêm chính sách chuyển bài chưa chắc chắn sang giảng viên xem lại. '
      'Không đặt mục tiêu điều chỉnh test để có F1 lớn nhất. Kết quả dưới đây là số đo thật của phiên Mac; không thay thế log Kaggle đã trình bày trước đó.')
    table(d,['Thư mục trong Project/Datasets','Số bài','Nguồn và vai trò'],[
        ['UPenn',3493,'Dữ liệu nghiên cứu gốc và bản đã xử lý'],
        ['Stanford',29070,'Dữ liệu nghiên cứu gốc và bản đã xử lý'],
        ['CustomForum',custom['rows'],'Kịch bản tổng hợp do trợ lý soạn để duyệt và khảo sát hành vi']], [6.4,2,8.6])
    p('Hai nguồn thật giữ nguyên nội dung, nhãn gốc và tập test lịch sử. Mỗi thư mục có raw.csv, processed.csv và metadata.json ghi nguồn cùng SHA256. '
      'Các đường dẫn data/raw và data/processed cũ là liên kết tương đối, nên chương trình cũ tiếp tục chạy. Không trộn bài tự viết vào tên nguồn UPenn hoặc Stanford.')
    table(d,['Loại tình huống CustomForum','Số bài'],[['Đang bị chặn và cần hỗ trợ',40],['Cùng vấn đề nhưng có cập nhật đã giải quyết',40],['Cập nhật nhưng vấn đề vẫn chưa giải quyết',40],
        ['Thiếu ngữ cảnh nên cần đọc thêm',12],['Ngoài phạm vi diễn đàn học tập',12],['Thông tin bình thường hoặc trích dẫn lời khẩn cấp',16]],[13,4])
    p(f"CustomForum có {custom['scenario_groups']} nhóm tình huống; chia cố định {custom['splits']['train']} train, {custom['splits']['validation']} validation và {custom['splits']['challenge']} challenge. "
      'Mọi biến thể của cùng tình huống luôn cùng một phần chia. draft_examples.csv chứa nhãn nháp; reviewed_labels.csv giữ phần duyệt riêng và không bị ghi đè khi sinh lại. '
      f"Hiện có {custom['human_approved_rows']} bài được duyệt; thí nghiệm dưới đây dùng {protocol['custom_rows_used']} bài custom.")
    p('Nhãn dự thảo CustomForum xét nhu cầu hỗ trợ hiện tại; nhãn benchmark gốc vẫn là urgency > 4. Hai khái niệm có thể khác nhau. '
      'Không coi nhãn do mô hình tự tạo là bằng chứng mô hình dự đoán đúng. Muốn mở rộng dữ liệu thật, cần thu thập bài đã ẩn danh, '
      'nhờ người đọc gán nhãn độc lập, giải quyết bất đồng và giữ một tập đích mới chưa dùng để sửa mô hình. ANNOTATION_GUIDE.md ghi cách duyệt.')
    page('DeBERTa và quy trình tách dữ liệu')
    p('Dùng microsoft/deberta-v3-base của Microsoft, revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3. '
      'Mô hình có 12 lớp, hidden size 768 và khoảng 184 triệu tham số gồm embedding. Giới hạn đầu vào là 512 token, so với 256 token của MiniLM fine-tuned trước đây. '
      'Đây là fine tuning mô hình pretrained trên nhãn bài đăng; không pretrain một mô hình ngôn ngữ từ đầu. '
      'Một câu vẫn không cung cấp trạng thái luồng hoặc lịch sử mà người nhập không đưa vào.')
    table(d,['Nguồn','Fit','Calibration','Validation','Test'],[[source,*[str(v['posts'])+' / '+str(v['urgent']) for v in roles.values()]]
        for source,roles in plan['source_splits'].items()],[3,3.5,3.5,3.5,3.5])
    p('Mỗi ô ghi tổng số bài / số bài ưu tiên. Fit dùng cập nhật trọng số; calibration dùng học phép Platt chuyển điểm sang ước lượng xác suất; '
      'validation chọn checkpoint và ngưỡng; test chỉ dùng so sánh. UPenn tách student ID ở cả bốn vai trò. Stanford không có ID tác giả hoặc luồng nên chỉ chia phân tầng. '
      'Các test lịch sử đã được xem ở phần trước, vì vậy đây là so sánh phát triển mang tính exploratory, chưa phải kiểm định ngoài hoàn toàn mới.')
    p(f"Huấn luyện chung {sum(protocol['source_fit_counts'].values())} bài thật, dùng trọng số loss để hai nguồn có đóng góp ngang nhau; không sao chép hàng hoặc tự đổi tỷ lệ lớp. "
      f"Batch {config['batch_size']}, tích lũy {config['accumulation']} batch, learning rate {config['lr']}, tối đa {config['epochs']} epoch, patience {config['patience']}, "
      f"seed {config['seed']}, thiết bị {protocol['device']} và {protocol['precision']}. Checkpoint theo trung bình AP validation của hai nguồn. "
      'Ngưỡng nhị phân theo balanced accuracy validation; không tối ưu F1 test. Phép calibration nguồn riêng và hỗn hợp được lưu riêng; app mặc định dùng hỗn hợp.')
    figure(d,'deberta_training.png','Hình 25. Loss train và AP validation của lần huấn luyện DeBERTa thật; history.json ghi từng epoch.')
    p(f"Lần chạy hoàn tất ghi {protocol['training_seconds']/60:.1f} phút gồm phần học và đánh giá. Lần FP32 trước đó dừng vì áp lực bộ nhớ trước validation đầu tiên, "
      'không có số đo chất lượng; fp32_runtime_attempt.json giữ sự kiện này. Phiên BF16 batch 4 giúp chạy trên Mac. '
      'Notebook Kaggle dùng một T4 với FP16 và gradient scaling; chưa có kết quả Kaggle của DeBERTa trong báo cáo này.')
    page('Kết quả nhị phân trên toàn bộ test lịch sử')
    table(d,['Test','P','R','F1','AP','Brier'],[[s,*[f(r[k]) for k in ['precision','recall','f1','average_precision','brier_score']]] for s,r in mixed.items()],[3,2.8,2.8,2.8,2.8,2.8])
    p('Các chỉ số trên dùng policy mixed giống khi nhập câu mới và tính trên mọi bài test, kể cả bài có hành động REVIEW. '
      'Precision và Recall xét lớp ưu tiên; Brier đo sai số của ước lượng xác suất, càng nhỏ càng tốt. '
      'Không dùng accuracy của riêng phần tự động làm accuracy của toàn hệ thống.')
    figure(d,'deberta_comparison.png','Hình 26. So với MiniLM và blend trên cùng test. DeBERTa huấn luyện chung hai nguồn; cấu hình học khác nên không cô lập tác động kiến trúc.')
    for source,r in mixed.items():
        tn,fp=r['confusion_matrix'][0];fn,tp=r['confusion_matrix'][1]
        p(f"{source}: tìm đúng {tp}/{r['positive_count']} bài ưu tiên, bỏ sót {fn}; {fp} bài thường bị báo ưu tiên. "
          f"Accuracy toàn test {percent(r['accuracy'])}, macro F1 {f(r['macro_f1'])}, ROC AUC {f(r['roc_auc'])}, P@20 {f(r['precision_at_20'])}. "
          'Đây là kết quả so nhãn dữ liệu hiện có, không phải xác suất đúng khi nhập một câu bất kỳ.')
    page('Ảnh hưởng của ngưỡng với lỗi phân loại')
    fixed=load(OUT/'fixed_threshold_diagnostic.json')
    p('Để tách ảnh hưởng của ngưỡng khỏi năng lực biểu diễn, chúng tôi còn tính lại các nhãn từ cùng điểm calibrated với ngưỡng cố định 0,5. '
      'Đây là phép chẩn đoán chuẩn, không tìm ngưỡng bằng test và không thay policy Review đã chọn. Không huấn luyện thêm trọng số trong bước này.')
    table(d,['Test','P tại 0,5','R tại 0,5','F1 tại 0,5','Accuracy'],[[r['source'],*[f(r[k]) for k in ['precision','recall','f1','accuracy']]] for r in fixed],[3,3.5,3.5,3.5,3.5])
    for row in fixed:
        fp=row['confusion_matrix'][0][1];fn=row['confusion_matrix'][1][0]
        p(f"{row['source']} tại 0,5 có {fp} báo ưu tiên sai và {fn} bỏ sót. "
          'Đổi ngưỡng làm thay đổi đánh đổi giữa hai loại lỗi; nội dung mô hình hiểu và thứ hạng điểm không thay đổi.')
    p('Ngưỡng balanced accuracy 0,07 tăng Recall nên F1 và độ đúng tổng thể có thể giảm vì nhiều báo động sai. '
      'Ở ngưỡng 0,5, Stanford ít báo động sai hơn nhưng bỏ sót nhiều bài ưu tiên hơn. '
      'Không nên đổi sang ngưỡng có con số test đẹp nhất rồi tuyên bố mô hình hiểu ngữ cảnh tốt hơn. '
      'Quy trình cần chọn chi phí bỏ sót, báo nhầm và khả năng đọc bài Review với giảng viên, rồi xác nhận trên bài thật mới.')
    table(d,['Test','Policy','Ngưỡng','Balanced accuracy validation','F1 test'],[[r['test_source'],r['policy'],f(r['threshold']),
        f(load(ROOT/'models/urgency_deberta_real_only/decision_config.json')['policies'][r['policy']]['validation_balanced_accuracy']),f(r['f1'])] for r in results],[3,3,3,4.5,3.5])
    p('Policy theo nguồn là phân tích chẩn đoán khi biết nguồn đầu vào; không chọn lại policy bằng dòng test tốt nhất. '
      'Hiệu quả có thể khác giữa hai nguồn do tỷ lệ lớp, chủ đề và cách gán nhãn. Không cam kết đạt 80% F1 hoặc đúng mọi câu.')
    p('Lượt DeBERTa hiện tại chưa chứng minh thay thế tốt hơn cả MiniLM và blend trên cả hai nguồn. '
      'Chúng tôi giữ model cũ và ghi cả kết quả kém. Model mới là ứng viên để khảo sát ngữ cảnh và chính sách Review, '
      'chưa phải một bản nâng cấp được xác nhận bằng dữ liệu thực địa độc lập.')
    page('Ma trận nhầm lẫn và hiệu chỉnh điểm')
    figure(d,'deberta_confusion.png','Hình 27. Ma trận [[TN FP] [FN TP]] của policy chung; mọi mẫu test được giữ trong phép tính.')
    figure(d,'deberta_calibration.png','Hình 28. Calibration curve theo tám bin có số mẫu gần bằng nhau. Đường chéo biểu diễn mức phù hợp lý tưởng.')
    p('Calibration học từ phần dữ liệu riêng; đường cong ở đây dùng test để kiểm tra sau huấn luyện. '
      'Điểm 0,9 vẫn có thể đi kèm dự đoán sai, nhất là khi gặp kiểu câu khác dữ liệu học. '
      'Calibration không bổ sung bối cảnh còn thiếu và không chứng nhận độ đúng cho mỗi câu. Ngưỡng gán nhãn nhị phân cũng không phải ngưỡng chắc chắn đúng.')
    p('Đầu vào dài hơn 512 token vẫn bị cắt; cần đưa phần cập nhật giải quyết và thông tin quan trọng vào đoạn mô hình đọc được. '
      'Model chưa có bộ phát hiện ngữ nghĩa ngoài phạm vi đã được kiểm định. Những câu như hỏi thời tiết hoặc một yêu cầu rất mơ hồ '
      'có thể vẫn nhận điểm cao hoặc thấp. Phạm vi sử dụng được đánh giá là bài đăng diễn đàn học tập tiếng Anh.')
    p('Các tệp predictions_upenn_mixed.csv và predictions_stanford_mixed.csv giữ post ID, nhãn, điểm, dự đoán và hành động cho từng bài. '
      'Có thể đếm lại TP FP FN TN hoặc tìm lỗi từ các tệp này; results.json giữ cả policy nguồn riêng. '
      'data_validation.json xác nhận SHA256 nguồn, test membership, tách student và nhóm tình huống.')
    page('Cơ chế giảng viên xem lại và các lỗi tự động')
    policy=load(ROOT/'models/urgency_deberta_real_only/decision_config.json')['policies']['mixed']
    low=min(policy['review_low'],policy['threshold']);high=max(policy['review_high'],policy['threshold'])
    p(f'Với policy chung của lượt này, bài có điểm ≤ {low:.4f} được tự xếp NORMAL và điểm ≥ {high:.4f} được tự xếp PRIORITY. '
      'Khoảng giữa là REVIEW. Giới hạn ban đầu 0,15 và 0,85 được siết để nhất quán với ngưỡng nhị phân validation. Đây là quy tắc hỗ trợ thao tác do dự án đặt trước, '
      'chưa có bảo đảm về tỷ lệ lỗi thống kê. Giảng viên vẫn cần đọc các bài REVIEW; chúng không được coi là đã xử lý đúng.')
    table(d,['Test','Tự động','Review','Lỗi tự động','Đúng trong phần tự động'],[[s,str(a['automatic_count'])+'/'+str(a['total']),a['review_count'],a['automatic_errors'],percent(a['automatic_accuracy'])] for s,a in audit.items()],[2.5,3.5,2.5,3,5.5])
    table(d,['Test','Ưu tiên tự tìm đúng','Ưu tiên tự bỏ sót','Ưu tiên vào Review','Thường bị tự báo ưu tiên'],[[s,a['urgent_automatically_found'],a['urgent_automatically_missed'],a['urgent_requiring_review'],a['false_urgent_automatic']] for s,a in audit.items()],[2.5,3.5,3.5,3.5,3.5])
    figure(d,'deberta_review.png','Hình 29. Đếm toàn bộ bài test thành tự động đúng, tự động sai và cần giảng viên xem lại; không ẩn mẫu khó.')
    p('Tỷ lệ tự động đúng có thể cao vì nhiều bài bình thường dễ nhận biết. Vì vậy reliability_audit.json còn ghi Precision ưu tiên tự động '
      'và tỷ lệ tìm được trên tổng số bài ưu tiên, cùng số bài ưu tiên đang chờ Review. Không dùng chỉ một con số để tuyên bố độ tin cậy. '
      'Trong CSV xuất từ app, cột action giữ riêng REVIEW, PRIORITY và NORMAL; nhãn nhị phân chỉ là gợi ý phụ khi action là REVIEW.')
    page('Khảo sát tình huống có thay đổi ngữ cảnh')
    figure(d,'deberta_context_drafts.png','Hình 30. Điểm của các cặp tổng hợp trước và sau cập nhật đã giải quyết. Đây là khảo sát hành vi trên nhãn nháp, không phải accuracy benchmark.')
    diagnostics=load(OUT/'custom_context_diagnostics.json')
    for name,counts in diagnostics.items():p(f"{name}: điểm giảm sau cập nhật đã giải quyết ở {counts['score_decreases_after_resolution']}/{counts['paired_scenarios']} cặp. "
      'Điểm giảm chưa chứng minh nhãn cuối đúng; chỉ cho thấy mô hình phản ứng với thay đổi nội dung.')
    examples=read(OUT/'custom_draft_predictions.csv');pair=[r for r in examples if r['scenario_group']=='submission_button']
    for row in pair:
        p(row['text'])
        p(f"Nhãn nháp {row['draft_action']}; DeBERTa score {float(row['deberta_score']):.4f}, action {row['deberta_action']}. "
          'Chưa có đánh giá độc lập xác nhận nhãn nháp này.')
    p('Ở nhóm ví dụ nộp bài trên, DeBERTa vẫn xếp ưu tiên sau cập nhật đã giải quyết; MiniLM cũng còn xếp ưu tiên. '
      'Điều này là lỗi theo nhãn dự thảo của tình huống và cho thấy việc đổi sang encoder lớn hơn chưa xử lý được ví dụ phủ định nhu cầu hỗ trợ. '
      'Cần người đọc xác nhận nhãn và bổ sung dữ liệu tương phản phù hợp trước khi kết luận cải thiện ngữ cảnh.')
    p('Bộ này được soạn để kiểm tra trạng thái đã giải quyết, từ khóa được trích dẫn, bài thiếu ngữ cảnh và nội dung ngoài phạm vi. '
      'Mẫu tự viết dễ có văn phong và cấu trúc đơn giản hơn bài thật. Dùng dữ liệu này để đạt điểm cao không chứng minh khả năng dùng trong lớp học. '
      'Sau khảo sát, thêm 40 cập nhật vẫn đang bị chặn để hạn chế shortcut Update đồng nghĩa đã giải quyết; các ID cũ giữ nguyên và nhãn vẫn chờ duyệt. Nhánh DeBERTa custom chỉ nhận nhãn đã duyệt. Thử nghiệm MiniLM cuối dùng weak supervision tổng hợp được ghi riêng; không dùng challenge để chọn checkpoint.')
    page('Lỗi thật và cách chạy lại sản phẩm')
    p('Dưới đây lấy tối đa hai lỗi tự động có điểm chắc chắn nhất của mỗi nguồn, ưu tiên một FP và một FN. '
      'Đoạn trích giữ nguyên nội dung đầu bài; bản đầy đủ có trong CSV dự đoán. Cột nhãn phản ánh nhãn benchmark, vẫn có thể có bất đồng giữa người đọc.')
    for source in ['upenn','stanford']:
        rows=read(OUT/f'predictions_{source}_mixed.csv')
        for label in ['0','1']:
            errors=[r for r in rows if r['label']==label and r['label']!=r['predicted'] and r['action']!='REVIEW']
            if not errors:continue
            row=max(errors,key=lambda r:abs(float(r['score'])-.5));excerpt=row['text'][:230]+('…' if len(row['text'])>230 else '')
            p(f"{source} nhãn {row['label']}, dự đoán {row['predicted']}, điểm {float(row['score']):.4f}: {excerpt}")
    p('Sai tự tin cho thấy mô hình có thể dựa vào cách diễn đạt, chủ đề hoặc thiếu ngữ cảnh khi gán mức khẩn cấp. '
      'Các lỗi không tự động chứng minh nhãn gốc sai. Cần đọc cả bài, trạng thái luồng và nhờ giảng viên đối chiếu trước khi bổ sung nhãn mới.')
    p('Model tự chứa ở models/urgency_deberta_real_only gồm trọng số, tokenizer, decision_config.json, requirements.txt và predict.py. '
      'Đã kiểm tra CPU offline với cache trống. Trong app chọn UPenn + Stanford · DeBERTa joint + review. '
      'Mô hình mặc định trước đây vẫn giữ để so sánh; candidate mới không được tuyên bố đã thay thế mọi model cũ.')
    p('Chạy app: bash Project/scripts/launchers/start_app.sh. Tạo lại mọi hình, ảnh và cùng Word từ kết quả đã lưu: bash Project/scripts/launchers/reproduce.sh. '
      'Train lại DeBERTa: bash Project/scripts/launchers/reproduce.sh --reliable. Notebook kaggle/reliable_deberta_training.ipynb mặc định huấn luyện thật, '
      'tự tải hai dữ liệu gốc và tạo folder ba dữ liệu. Bật Internet và GPU T4; không cần thêm dataset cho lần chạy real only. '
      'Khi có nhãn custom đã duyệt, dùng include approved custom để tạo run riêng; không ghi đè run real only.')
    p('Nguồn kiến trúc: https://huggingface.co/microsoft/deberta-v3-base. Dữ liệu gốc: https://github.com/pcla-code/forum-posts-urgency. '
      'Giải thích calibration: https://scikit-learn.org/stable/modules/calibration.html. model_source.json và protocol.json ghi revision cố định.')
    shot=R/'screenshots/10_deberta_review.png'
    if shot.exists():
        page('Ứng dụng chạy thật với quyết định xem lại')
        para=d.add_paragraph();para.paragraph_format.keep_with_next=True;para.add_run().add_picture(str(shot),width=Cm(17))
        para=d.add_paragraph('Hình 31. Chrome nhận đúng hành động REVIEW từ API DeBERTa. browser_checks.json đối soát model, điểm, action và giao diện.')
        for run in para.runs:run.italic=True;run.font.size=Pt(8.5)
        p('Giao diện thể hiện chưa tự động xếp loại khi cần giảng viên xem lại. Điểm hiển thị là ước lượng, không phải mức bảo đảm đúng. '
          'Hàng đợi và CSV giữ cả hành động Review; nội dung bài không được gửi tới dịch vụ AI bên ngoài.')
