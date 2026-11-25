"""Append the measured fine-tuning extension to the same DOCX during regeneration."""
import csv,json,re
from pathlib import Path
from docx.shared import Cm,Pt
from build_report import table,figure

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';FT=R/'fine_tuning'
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def fmt(x):return f'{x:.3f}'

def append_fine_tuning(document):
    if not (FT/'results.json').exists():return
    fine=load(FT/'results.json');baseline=load(R/'experiments/results.json');protocol=load(FT/'protocol.json')
    sources=['upenn','stanford'];selection={s:load(FT/s/'selection.json') for s in sources};cv={s:load(FT/s/'cv.json') for s in sources}
    def p(text):document.add_paragraph(re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',' ',text))
    def h(text):document.add_heading(text,2)
    def page(text):document.add_page_break();document.add_heading(text,1)
    def code(text):
        para=document.add_paragraph(text)
        for run in para.runs:run.font.name='Courier New';run.font.size=Pt(8.5)
    def row(a,b,model):return next(r for r in baseline+fine if r['train_source']==a and r['test_source']==b and r['model']==model)
    def image(path,caption):
        para=document.add_paragraph();para.paragraph_format.keep_with_next=True;para.add_run().add_picture(str(path),width=Cm(17))
        para=document.add_paragraph(caption)
        for run in para.runs:run.italic=True;run.font.size=Pt(8.5)

    page('Thực nghiệm bổ sung fine tuning MiniLM cho nhãn khẩn cấp')
    p('Phần này mở rộng trực tiếp báo cáo gốc: giữ lại 20 kết quả trước đó và bổ sung bốn kết quả kiểm tra của classifier MiniLM fine-tuned. '
      'Toàn bộ encoder và head nhị phân được cập nhật bằng nhãn urgency, thay vì chỉ dùng embeddings cố định. '
      'Không pretrain ngôn ngữ từ đầu. Model khởi tạo từ cùng revision all-MiniLM-L6-v2 đã dùng trong thí nghiệm frozen.')
    table(document,['Thành phần','Cấu hình thực nghiệm'],[
        ['Seed và dữ liệu','42; dùng lại đúng split_ids.csv, test không tham gia tuning'],
        ['Optimizer và regularization','AdamW; learning rate 2e-5; weight decay 0,01; gradient clipping 1'],
        ['Batch và văn bản',f"Batch {protocol['batch_size']}; tối đa {protocol['max_length']} token; padding động theo nhóm độ dài"],
        ['Epoch và early stopping',f"Tối đa {protocol['max_epochs']} epoch; patience {protocol['patience']}; chọn checkpoint theo AP validation"],
        ['Precision tính toán','bfloat16 autocast trên MPS; trọng số lưu dạng float32'],
        ['Cross validation',f"{protocol['folds']} fold trong train nguồn; giữ nhóm sinh viên UPenn"],
        ['Fit cuối và ngưỡng','Median best epoch của 5 fold; ngưỡng tối đa F1 trên OOF nguồn']], [5.2,11.8])
    p('Ba cấu hình được thử trên cùng scout validation riêng của mỗi nguồn: cross-entropy thường; class-weighted loss; '
      'class-weighted loss cộng contraction expansion. Trọng số lớp được tính trên dữ liệu train của chính lượt fit. '
      'Cấu hình chọn theo F1 validation sau ngưỡng; nếu bằng nhau chọn AP cao hơn. Không dùng test để quyết định cấu hình thắng.')
    table(document,['Nguồn và cấu hình','Val AP','Val F1','Best epoch','Bài tăng thêm'],
        [[s+' '+r['name'].replace('scout_',''),fmt(r['validation_ap']),fmt(r['validation_f1_tuned']),r['best_epoch'],r['augmented_posts']]
         for s in sources for r in selection[s]['scout_runs']],[7,2.5,2.5,2.5,2.5])
    p('Mọi cấu hình đều được lưu, kể cả cấu hình kém. Ví dụ plain loss trên UPenn cho kết quả thấp hơn rõ rệt ở scout validation; '
      'không bỏ lượt chạy đó khỏi báo cáo. Scout là bước chọn cấu hình nên các điểm này không phải kết quả test độc lập.')

    page('Learning curves và chẩn đoán năm fold')
    figure(document,'ft_learning_curves.png','Hình 15. Loss train và AP validation của cả ba cấu hình ở hai nguồn; lấy từ history.json từng lượt chạy.')
    table(document,['Nguồn và fold','Best epoch','Val AP','Val F1 sau ngưỡng'],
        [[s+' fold '+str(i+1),r['best_epoch'],fmt(r['validation_ap']),fmt(r['validation_f1_tuned'])]
         for s in sources for i,r in enumerate(cv[s]['folds'])],[6,3,4,4])
    p('Mỗi fold khởi tạo lại pretrained checkpoint và seed. Validation fold được dùng để chọn checkpoint theo AP và theo dõi early stopping; '
      'điểm OOF được dùng chọn ngưỡng. Do đã tham gia model selection và early stopping, đây là development diagnostics, '
      'không phải nested CV độc lập hay bằng chứng thống kê chắc chắn về ưu thế mô hình.')
    stopped=sum(r['epochs_run']<protocol['max_epochs'] for s in sources for r in selection[s]['scout_runs']+cv[s]['folds'])
    p(f"Có {stopped} lượt scout/fold dừng trước giới hạn epoch. Các lượt chạy tới giới hạn vẫn giữ checkpoint tốt nhất theo validation, "
      'không mặc định lấy epoch cuối. Báo cáo không diễn giải early stopping như một thí nghiệm đối chứng riêng vì tất cả lượt fine-tuning đều bật cùng quy tắc.')

    page('Độ biến thiên giữa các fold và ngưỡng OOF')
    figure(document,'ft_cv.png','Hình 16. F1 validation sau tuning ngưỡng của năm fold; đây là điểm phát triển, không phải test độc lập.')
    table(document,['Nguồn','Mean fold F1','Std fold F1','OOF F1','Ngưỡng','Epoch fit cuối'],
        [[s,fmt(cv[s]['fold_f1_mean']),fmt(cv[s]['fold_f1_std']),fmt(cv[s]['oof_metrics']['f1']),
          fmt(cv[s]['threshold']),cv[s]['epochs_for_final']] for s in sources],[3.5,3,3,2.5,2.5,2.5])
    p('Mean fold F1 lấy trung bình F1 đã tune ngưỡng riêng của từng fold. OOF F1 gộp điểm của mọi bài train rồi chọn một ngưỡng nguồn chung; '
      'hai giá trị có thể khác nhau vì F1 không tuyến tính và quy tắc ngưỡng khác nhau. App dùng ngưỡng OOF chung đã lưu, không biết bài mới thuộc fold nào.')
    p('Fold IDs được lưu trong cv_fold_ids.csv. UPenn giữ sinh viên nguyên nhóm; Stanford phân tầng theo nhãn vì không có ID tác giả/luồng. '
      'Mọi lần fit đều tạo model mới từ pretrained, không tiếp tục từ trọng số fold trước. Biến thể augmentation chỉ sinh sau khi chia train/validation của fold.')
    p('Số epoch fit cuối là trung vị làm tròn từ best epoch của năm fold, sau đó học lại toàn bộ train nguồn. '
      'Lượt fit cuối không có early stopping vì không giữ thêm validation; số epoch đã được quyết định trước từ CV. Test không tham gia bước này.')
    p('Các điểm CV có thể lạc quan do scout selection, checkpoint selection và threshold tuning đã sử dụng dữ liệu phát triển. '
      'Một phép đánh giá nested CV hoặc một tập ngoài hoàn toàn mới sẽ cần thiết nếu muốn báo cáo ước lượng tổng quát hóa ít thiên lệch hơn.')

    page('So sánh classifier fine tuned với các pipeline trước')
    table(document,['Chiều và pipeline','Precision','Recall','F1','AP'],
        [[a+' → '+b+' '+name,*[fmt(row(a,b,model)[k]) for k in ['precision','recall','f1','average_precision']]]
         for a,b in [('upenn','upenn'),('stanford','stanford'),('upenn','stanford'),('stanford','upenn')]
         for model,name in [('tfidf_logreg','TF-IDF LR'),('minilm_logreg','frozen LR'),('finetuned_minilm','fine-tuned')]],[7.2,2.45,2.45,2.45,2.45])
    figure(document,'ft_comparison.png','Hình 17. So sánh trên cùng test: TF-IDF + LR, MiniLM frozen + LR và classifier fine-tuned.')
    for source in sources:
        current=row(source,source,'finetuned_minilm');old=row(source,source,'tfidf_logreg');frozen=row(source,source,'minilm_logreg')
        p(f"{source}: fine-tuned F1 = {fmt(current['f1'])}; chênh với TF-IDF LR là {fmt(current['f1']-old['f1'])}, "
          f"chênh với frozen LR là {fmt(current['f1']-frozen['f1'])}. "
          'Chênh lệch đo cho toàn bộ pipeline: cách học biểu diễn, loss, CV và ngưỡng cùng thay đổi; chưa cô lập riêng tác động từng yếu tố.')

    page('Ma trận nhầm lẫn và chất lượng hàng đợi của model mới')
    figure(document,'ft_confusion.png','Hình 18. Ma trận nhầm lẫn của hai model fine-tuned trên bốn hướng; đọc từ test_results.json.')
    table(document,['Train → test','Macro F1','Accuracy','P@20','R@20'],
        [[r['train_source']+' → '+r['test_source'],*[fmt(r[k]) for k in ['macro_f1','accuracy','precision_at_20','recall_at_20']]] for r in fine],[6,3,3,2.5,2.5])
    p('Các chỉ số dùng cùng định nghĩa như phần trước: TP/FP/FN tính trên lớp urgent; P@20 và R@20 dựa trên điểm liên tục. '
      'ROC-AUC, AP, kết quả ở ngưỡng mặc định 0,5 và từng dự đoán cũng được lưu trong reports/fine_tuning. '
      'Ngưỡng triển khai được lấy từ OOF train nguồn, không đổi ngưỡng để làm đẹp kết quả chuyển nguồn.')
    p('Tập test đã được xem trong quá trình nghiên cứu ban đầu. Các so sánh mở rộng được xem là exploratory; '
      'không tuyên bố đây là xác nhận trên dữ liệu hoàn toàn mới. Muốn kết luận khả năng áp dụng thực tế cần dữ liệu diễn đàn đích hoặc một bộ kiểm tra mới chưa tham gia phát triển.')

    page('Phân tích augmentation và lỗi còn lại sau fine tuning')
    figure(document,'ft_pr.png','Hình 19. Precision–Recall của classifier fine-tuned và hai pipeline LR trên test trong nguồn.')
    for source in sources:
        cards={r['name'].replace('scout_',''):r for r in selection[source]['scout_runs']}
        delta=cards['weighted_aug']['validation_f1_tuned']-cards['weighted']['validation_f1_tuned']
        p(f"Scout {source}: contraction augmentation thay đổi F1 validation so với weighted-only {delta:+.3f}; "
          f"cấu hình chọn là {selection[source]['selected']}. Chỉ thêm một biến thể khi mở rộng contractions làm văn bản thay đổi. "
          'Không sửa deadline hoặc bỏ phủ định, không tăng mẫu validation/test, không để bản gốc và biến thể ở hai fold khác nhau.')
    for source in sources:
        with (FT/source/f'predictions_{source}.csv').open(encoding='utf-8',newline='') as handle:predictions=list(csv.DictReader(handle))
        wrong=[r for r in predictions if r['label']!=r['predicted']]
        p(f"{source}: còn {len(wrong)} dự đoán sai trên test. Ví dụ dưới lấy theo thứ tự post_id, không chọn để tạo ấn tượng rằng lỗi đã được giải quyết hết.")
        for r in wrong[:2]:
            p(f"{r['post_id']} | thật {r['label']}, dự đoán {r['predicted']}, điểm {float(r['score']):.3f}: “"+
              r['text'][:180]+('…' if len(r['text'])>180 else '')+'”')
    p('Giới hạn còn lại gồm nhãn chủ quan, thiếu bối cảnh luồng, cắt bài vượt 256 token và domain shift. '
      'Không tự sửa nhãn test khi nội dung có vẻ mâu thuẫn. Augmentation ở đây có phạm vi nhỏ; chưa chứng minh paraphrasing, dịch vòng hoặc augmentation bằng LLM sẽ có lợi.')

    page('Thư mục model độc lập và notebook Kaggle có thể chạy lại')
    p('Mỗi nguồn có một model độc lập trong models/urgency_minilm_upenn hoặc models/urgency_minilm_stanford. '
      'Thư mục chứa weights safetensors, config, tokenizer, decision_config.json, predict.py và requirements.txt. '
      'Có thể sao chép cả thư mục sang project khác; predict.py chứa đúng hàm chuẩn hóa văn bản, không phụ thuộc prepare_data.py bên ngoài.')
    code('bash -c \'source Project/scripts/launchers/env.sh; Project/.venv/bin/python \\\nProject/models/urgency_minilm_stanford/predict.py \\\n--text "I cannot submit my assignment. Please help!"\'')
    p('Kiểm tra standalone dùng CPU float32, bật chế độ offline và trỏ HF cache tới một thư mục rỗng. '
      'Việc reload model và dự đoán hai câu mới đã được kiểm tra riêng. Điểm float32 CPU có thể lệch nhẹ so với bfloat16 MPS dùng trong lượt đánh giá; '
      'không hứa số học bit-identical giữa thiết bị.')
    nb=FT/'notebook_validation.json'
    if nb.exists():
        validation=load(nb)
        p(f"Notebook kaggle/urgency_minilm_training.ipynb đã chạy kiểm tra đủ {validation['executed_code_cells']} code cell trên máy local, "
          'tái dùng các lượt training thực tế đã hoàn thành. Các output hiển thị ghi rõ chế độ này. '
          'Mặc định khi upload vào Kaggle và Run All, notebook sẽ train lại. '
          + ('Log phiên Kaggle do người dùng cung cấp được đối chiếu riêng trong phần bổ sung cuối báo cáo; '
             'không thay output kiểm tra local bằng số của phiên đó.' if (R/'kaggle_run/results.json').exists() else
             'Chưa xác nhận bằng một phiên Kaggle GPU thực tế.'))
    p('Notebook tự chứa mã nguồn, tải hai CSV theo commit/hash cố định, dựng lại baseline và tập chia, chạy scout/5-fold/final fit, '
      'hiển thị toàn bộ bảng chỉ số và hình, phân tích ví dụ sai, cho phép nhập câu mới và xuất ZIP model cùng thống kê. '
      'Trên Kaggle cần bật Internet và GPU. Các phiên bản thư viện và thiết bị được in để giải thích sai khác có thể có khi chạy khác môi trường.')
    code('bash Project/scripts/launchers/reproduce.sh --finetune\n# Chạy lại fine tuning và cập nhật cùng báo cáo Word\nbash Project/scripts/launchers/reproduce.sh\n# Chỉ tạo lại biểu đồ và báo cáo từ kết quả đã lưu')
    total=sum(r['seconds'] for s in sources for r in selection[s]['scout_runs']+cv[s]['folds'])
    p(f"Tổng thời gian đo của scout và 5-fold ở hai nguồn là khoảng {total/60:.1f} phút, chưa gồm fit cuối, tải model, vẽ hình hoặc tạo Word. "
      'Notebook có thể chạy lâu hơn hoặc nhanh hơn tùy GPU; không dùng thời gian 2–4 phút của pipeline frozen để ước lượng full fine-tuning.')

    screenshot=R/'screenshots/07_finetuned.png'
    if screenshot.exists():
        page('Minh chứng dùng model fine tuned trong ứng dụng hiện có')
        image(screenshot,'Hình 20. Chrome chạy thật với lựa chọn Stanford MiniLM fine-tuned trong cùng ứng dụng.')
        checks=load(R/'screenshots/browser_checks.json');entry=next((c for c in checks if c['case']=='finetuned_urgent'),None)
        if entry:p(f"Câu minh họa trả nhãn {entry['label']}, điểm {entry['score']:.4f}, ngưỡng {entry['threshold']:.4f}; API đã được đối soát với giao diện.")
        p('Dropdown vẫn giữ các mô hình trước để so sánh trực tiếp, và thêm hai lựa chọn fine-tuned. '
          'Mặc định ban đầu TF-IDF + LR được giữ nguyên; người dùng chọn MiniLM fine-tuned để dùng model mới. '
          'Trang kết quả hiển thị thêm bốn kết quả test mới. Việc mô hình nào đứng đầu bảng test là mô tả hậu nghiệm, không thay đổi quyết định chọn cấu hình theo validation.')
        p('Model mới cải thiện trong một số hướng đánh giá nhưng chưa thay thế việc đọc bài của giảng viên. '
          'Toàn bộ scout thất bại, learning curves, fold IDs và dự đoán test vẫn được giữ để giảng viên kiểm tra quá trình thực nghiệm.')
