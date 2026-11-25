"""Create the concise official report from every completed experiment family."""
import csv,json,re
from pathlib import Path
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from build_report import canonical_save
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';F=R/'final_minilm_trial'
NAMES={'tfidf_logreg':'TF-IDF + LR','tfidf_svm':'TF-IDF + SVM','tfidf_nb':'TF-IDF + NB','tfidf_dummy':'Đa số',
       'minilm_logreg':'MiniLM frozen + LR','finetuned_minilm':'MiniLM fine tuned','minilm_tfidf_blend':'MiniLM + TF-IDF',
       'joint_deberta':'DeBERTa joint','real_only':'B real-only','custom_ce':'C custom CE','custom_pairwise':'C custom pair'}
def load(path):return json.loads(path.read_text())
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def num(v):return '—' if v is None else f'{v:.4f}'
def cm(r):return f"{r['confusion_matrix'][0][0]}, {r['confusion_matrix'][0][1]}; {r['confusion_matrix'][1][0]}, {r['confusion_matrix'][1][1]}"
def setting(r):
    names={'upenn':'U','stanford':'S','joint':'J','upenn+stanford':'J','custom_challenge':'T*'}
    return names.get(r.get('train_source','joint'),r.get('train_source','J'))+'→'+names.get(r['test_source'],r['test_source'])
def main():
    baseline=load(R/'experiments/results.json');fine=load(R/'fine_tuning/results.json');blend=load(R/'ensembles/results.json')
    large=load(R/'reliability/deberta_real_only/results.json');fixed=load(R/'reliability/deberta_real_only/fixed_threshold_diagnostic.json')
    kaggle=load(R/'kaggle_run/results.json');kb=load(R/'kaggle_run/baseline_results_from_log.json')
    joint_kaggle=load(R/'kaggle_run/joint_minilm_20261117_results.json')
    ai_audit=load(ROOT/'Datasets/CustomForum/ai_label_audit_20261117.json')
    final=load(F/'summary.json');tests=final['results'];plan=load(F/'selection.json');seeds=load(F/'two_seed_summary.json')
    probes=load(F/'behavior_probe_results.json');errors=load(F/'error_examples.json')
    selected=plan['selected_custom_configuration'];candidate=selected+'_seed42';control='real_only_seed42'
    audit=load(R/'data_audit.json');split=load(R/'reliability/data_plan.json');custom=load(ROOT/'Datasets/CustomForum/metadata.json')
    modelsize=load(R/'reliability/model_comparison_discussion.json');history=load(R/'reliability/deberta_real_only/history.json')
    d=Document();s=d.sections[0];s.top_margin=Cm(1.8);s.bottom_margin=Cm(1.7);s.left_margin=s.right_margin=Cm(2)
    s.page_width=Cm(21);s.page_height=Cm(29.7)
    normal=d.styles['Normal'];normal.font.name='Arial';normal.font.size=Pt(10)
    normal.paragraph_format.space_after=Pt(6);normal.paragraph_format.line_spacing=1.1
    for name,size in [('Title',20),('Subtitle',12),('Heading 1',16),('Heading 2',11)]:
        style=d.styles[name];style.font.name='Arial';style.font.size=Pt(size);style.font.color.rgb=RGBColor.from_string('000000')
        for border in list(style.element.iter(qn('w:pBdr'))):border.getparent().remove(border)
    for name in ['Heading 1','Heading 2']:d.styles[name].paragraph_format.keep_with_next=True
    foot=s.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)
    pages=[];all_rows=[]
    def p(text,style=None):
        text=display_name(text)
        safe=''.join(ch if ch in '\t\n\r' or 32<=ord(ch)<=0xD7FF or 0xE000<=ord(ch)<=0xFFFD or 0x10000<=ord(ch)<=0x10FFFF else f'[U+{ord(ch):04X}]' for ch in text)
        return d.add_paragraph(safe,style)
    def heading_text(text):return re.sub(r'\s+',' ',re.sub(r'[^\w\s]',' ',text)).strip()
    def h(text):d.add_heading(heading_text(text),2)
    def model_legend(include_ce=True):
        text='MiniLM-B và các biến thể MiniLM-C đều tinh chỉnh all-MiniLM-L6-v2 cho phân loại ưu tiên. MiniLM-B học chung UPenn và Stanford. '
        if include_ce:
            text+='MiniLM-C-CE học thêm dữ liệu tổng hợp theo nhãn từng câu; MiniLM-C-Pair học thêm cả thứ tự ưu tiên giữa các câu cùng tình huống.'
        else:
            text+='MiniLM-C-Pair học thêm dữ liệu tổng hợp theo nhãn từng câu và thứ tự ưu tiên giữa các câu cùng tình huống.'
        p(text)
    def display_name(value):
        text=str(value)
        for old,new in [('B real-only','MiniLM-B'),('C custom CE','MiniLM-C-CE'),('C custom pair','MiniLM-C-Pair')]:
            text=text.replace(old,new)
        for old,new in [('MiniLM-C-Pair','MiniLM + học cặp'),('MiniLM-C-CE','MiniLM + CE tổng hợp'),('MiniLM-B','MiniLM hai nguồn'),('các biến thể MiniLM-C','các phương án thêm dữ liệu tổng hợp')]:
            text=text.replace(old,new)
        return text
    def page(title):
        if pages:d.add_page_break()
        title=heading_text(title)
        if pages:d.add_heading(title,1)
        else:p(title,'Title')
        pages.append(title)
    def table(headers,records,widths=None,font=8.4):
        t=d.add_table(rows=1,cols=len(headers));t.autofit=False
        borders=OxmlElement('w:tblBorders')
        for side in ['top','left','bottom','right','insideH','insideV']:
            border=OxmlElement('w:'+side);border.set(qn('w:val'),'single');border.set(qn('w:sz'),'4');border.set(qn('w:color'),'D9D9D9');borders.append(border)
        t._tbl.tblPr.append(borders)
        widths=widths or [17/len(headers)]*len(headers)
        for column,width in zip(t.columns,widths):column.width=Cm(width)
        for i,v in enumerate(headers):t.rows[0].cells[i].text=display_name(v)
        repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for record in records:
            cells=t.add_row().cells
            for cell,value in zip(cells,record):cell.text=display_name(value)
        for j,row in enumerate(t.rows):
            prevent=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(prevent)
            for i,cell in enumerate(row.cells):
                cell.width=Cm(widths[i]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                pr=cell._tc.get_or_add_tcPr();margins=OxmlElement('w:tcMar')
                for side,value in [('top',45),('bottom',45),('left',75),('right',75)]:
                    node=OxmlElement('w:'+side);node.set(qn('w:w'),str(value));node.set(qn('w:type'),'dxa');margins.append(node)
                pr.append(margins)
                if j==0 or j%2==0:
                    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E6EEF4' if j==0 else 'F5F7F9');pr.append(shade)
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_after=Pt(1);paragraph.paragraph_format.line_spacing=1.04
                    for run in paragraph.runs:run.font.name='Arial';run.font.size=Pt(font);run.bold=j==0
        p('').paragraph_format.space_after=Pt(1)
        return t
    def picture(path,caption,width=16):
        paragraph=p('');paragraph.paragraph_format.keep_with_next=True
        paragraph.add_run().add_picture(str(path),width=Cm(width))
        paragraph=p(caption)
        for run in paragraph.runs:run.italic=True;run.font.size=Pt(8)
    def metrics_table(records,environment,label=None):
        output=[]
        for r in records:
            model=label(r) if label else NAMES.get(r['model'],r['model'])
            accuracy=r.get('accuracy')
            if accuracy is None:
                a=r['confusion_matrix'];accuracy=(a[0][0]+a[1][1])/sum(map(sum,a))
            output.append([setting(r),model,num(r.get('threshold')),*[num(r[k]) for k in ['precision','recall','f1']],num(accuracy),cm(r)])
            all_rows.append({'environment':environment,'trial_label':model,**r})
        return table(['Hướng','Mô hình / chính sách','Ngưỡng','P','R','F1','Acc','TN, FP; FN, TP'],output,[1.25,3.2,1.7,1.25,1.25,1.25,1.25,4.85],font=8)
    def result(arm,source,seed=42):return next(r for r in tests if r['arm']==arm and r['seed']==seed and r['test_source']==source)
    # 1
    page('ƯU TIÊN PHẢN HỒI BÀI ĐĂNG SINH VIÊN')
    p('BÁO CÁO ĐỒ ÁN HỌC MÁY', 'Subtitle')
    p('Thực nghiệm, phân tích sai số và sản phẩm ứng dụng • 17/11/2026')
    h('Mục tiêu và sản phẩm')
    p('Xây dựng ứng dụng hỗ trợ giảng viên xếp thứ tự bài đăng diễn đàn học tập cần phản hồi. Đầu vào là văn bản tiếng Anh; đầu ra gồm nhãn ưu tiên, điểm xếp hạng và đề nghị xem lại khi điểm chưa rõ. Sản phẩm web cho nhập câu mới, xử lý CSV và so sánh kết quả thực nghiệm.')
    h('Mô hình gốc và các phương pháp huấn luyện')
    p('Ba phương án dưới đây cùng dùng mô hình có sẵn sentence-transformers/all-MiniLM-L6-v2. Mô hình này chuyển văn bản thành biểu diễn số. Dự án gắn thêm bộ phân loại hai lớp và tinh chỉnh cả bộ mã hóa lẫn bộ phân loại bằng các bài đăng có nhãn. Cách này được gọi là học có giám sát kết hợp tinh chỉnh mô hình đã huấn luyện trước.')
    table(['Tên dùng trong báo cáo','Dữ liệu huấn luyện','Phương pháp'],[
        ['MiniLM hai nguồn','UPenn và Stanford; không dùng dữ liệu tổng hợp.','Tinh chỉnh MiniLM để phân loại cần ưu tiên / chưa cần ưu tiên bằng hàm mất mát entropy chéo (cross-entropy). Cân bằng tổng đóng góp của hai nguồn thật.'],
        ['MiniLM + CE tổng hợp','Hai nguồn thật và 82 câu tổng hợp có nhãn nhị phân trong tập huấn luyện.','Như phương án hai nguồn, thêm cross-entropy (CE) trên từng câu tổng hợp. Phần dữ liệu thật đóng góp 80% và phần tổng hợp đóng góp 20% vào hàm mất mát.'],
        ['MiniLM + học cặp','Cùng dữ liệu với phương án thêm CE tổng hợp; giữ các câu cùng tình huống thành nhóm.','Thêm yêu cầu câu còn cần hỗ trợ có điểm cao hơn câu đã giải quyết. Kết hợp CE với hàm mất mát xếp hạng theo cặp (pairwise ranking).']], [3.5,5.2,8.3],font=9)
    p('Ký hiệu B và C dùng để phân biệt các phương án huấn luyện trong dự án. B (real_only) tương ứng MiniLM hai nguồn; C-CE (custom_ce) tương ứng MiniLM + CE tổng hợp; C-Pair (custom_pairwise) tương ứng MiniLM + học cặp. Cả ba cùng xuất phát từ một mô hình MiniLM gốc, khác nhau ở dữ liệu và hàm mất mát khi tinh chỉnh.')
    h('Quy trình học và cách tạo dự đoán')
    p('Mỗi bài đăng được tách thành token, tối đa 256 token, rồi đưa qua MiniLM và bộ phân loại. Kết quả dự đoán được so với nhãn 0 hoặc 1 để tính sai số. Thuật toán AdamW cập nhật trọng số nhằm giảm sai số này; learning rate là 2e-5, batch size là 64. Các phương án được huấn luyện tối đa 4 epoch và chọn bộ trọng số theo AP trung bình trên tập xác thực của hai nguồn thật. Dữ liệu kiểm tra không được dùng để cập nhật trọng số.')
    p('Sau huấn luyện, điểm được hiệu chỉnh bằng Platt scaling trên tập hiệu chỉnh riêng. Ngưỡng nhị phân là 0,5; điểm lớn hơn 0,2 và nhỏ hơn 0,8 cần giảng viên xem lại. Phương án mặc định là MiniLM hai nguồn, seed 42. Cả ba phương án trên đều không kết hợp TF-IDF; MiniLM + TF-IDF blend là thí nghiệm riêng, kết hợp điểm của MiniLM và TF-IDF + Logistic Regression.')
    p('Các câu tổng hợp và nhãn của chúng do trợ lý soạn, chưa được con người duyệt. Những thí nghiệm thêm dữ liệu tổng hợp được trình bày riêng để so sánh với phương án chỉ dùng dữ liệu thật.')
    page('Tóm tắt kết quả thực nghiệm')
    h('Kết luận chính')
    table(['Kết luận','Bằng chứng đo được'],[
        ['Kết quả phụ thuộc phương pháp và dữ liệu','Tinh chỉnh MiniLM và kết hợp với TF-IDF giúp cải thiện một số kết quả. Dùng mô hình lớn hơn hoặc thêm dữ liệu tổng hợp không phải lúc nào cũng tốt hơn.'],
        ['Thêm dữ liệu tổng hợp chưa cải thiện kết quả trên dữ liệu thật','Mô hình thêm dữ liệu tổng hợp có F1 thấp hơn mô hình chỉ dùng dữ liệu thật trên cả hai nguồn và cả hai seed. Với 16 câu kiểm tra bổ sung, hai mô hình lần lượt trả lời đúng 13 và 14 câu.'],
        ['Dữ liệu tổng hợp giúp ở một số tình huống','Mô hình học thêm dữ liệu tổng hợp phân biệt đúng các nhóm tình huống trong bộ kiểm tra tổng hợp, nhưng vẫn sai với một số cách diễn đạt mới. Nhãn của bộ này chưa được con người duyệt.'],
        ['Mô hình được chọn cho ứng dụng','Ứng dụng dùng MiniLM học chung UPenn và Stanford. Các mô hình khác được giữ để so sánh. Kết quả hiện tại chưa chứng minh mô hình này tốt nhất trong thực tế; bài chưa chắc chắn vẫn cần giảng viên xem lại.']], [5,12],font=9)
    h('Kết quả so sánh hai phương án huấn luyện')
    model_legend(include_ce=False)
    table(['Phương án / seed 42','UPenn F1','Stanford F1','Đúng / 16 câu'],[
        ['MiniLM-B',num(result('real_only','upenn')['f1']),num(result('real_only','stanford')['f1']),'14 / 16'],
        ['MiniLM-C-Pair',num(result(selected,'upenn')['f1']),num(result(selected,'stanford')['f1']),'13 / 16']], [7,3,3,4])
    p('F1 đánh giá khả năng nhận diện bài cần ưu tiên trên toàn bộ tập kiểm tra, không phải tỷ lệ dự đoán đúng. Cả hai phương án dùng ngưỡng quyết định đã cố định. 16 câu bổ sung do trợ lý soạn và chưa có nhãn được con người xác nhận. Các tập kiểm tra thật đã được xem trong quá trình phát triển, nên cần thêm đánh giá độc lập trước khi kết luận về hiệu quả thực tế.')
    h('Phạm vi báo cáo')
    p('Báo cáo so sánh các mô hình cơ sở, MiniLM giữ nguyên trọng số, MiniLM được tinh chỉnh, phương án kết hợp với TF-IDF, DeBERTa và MiniLM học chung hai nguồn. Phần thực nghiệm cũng xem xét tăng cường dữ liệu, kiểm định chéo năm phần và kết quả chạy trên Kaggle. Nhật ký huấn luyện và điểm dự đoán từng bài được lưu kèm để đối chiếu.')
    page('Chi tiết 16 câu kiểm tra bổ sung')
    p('Bảng dưới so sánh MiniLM hai nguồn với MiniLM + học cặp. Cả hai đều tinh chỉnh all-MiniLM-L6-v2 và bộ phân loại; phương án học cặp bổ sung dữ liệu tổng hợp cùng yêu cầu xếp thứ tự ưu tiên. Hai mô hình lần lượt đúng 14/16 và 13/16 câu. Nội dung tiếng Anh được giữ nguyên như khi đánh giá. Nhãn 1 là cần ưu tiên, nhãn 0 là chưa cần ưu tiên. Nhãn kỳ vọng do trợ lý soạn, chưa được con người duyệt.')
    bprobe=next(r for r in probes if r['model_key']=='joint__minilm_real_control')
    cprobe=next(r for r in probes if r['model_key']=='joint__minilm_custom_trial')
    binary_probes=[r for r in bprobe['predictions'] if r['expected_binary_priority'] is not None]
    custom_by_id={r['id']:r for r in cprobe['predictions']}
    probe_rows=[]
    for i,row in enumerate(binary_probes,1):
        other=custom_by_id[row['id']]
        assert other['text']==row['text'] and other['expected_binary_priority']==row['expected_binary_priority']
        expected=row['expected_binary_priority']
        probe_rows.append([i,row['text'],expected,
            str(row['priority'])+(' — Đúng' if row['priority']==expected else ' — Sai'),
            str(other['priority'])+(' — Đúng' if other['priority']==expected else ' — Sai')])
    assert len(probe_rows)==16
    assert sum(r['priority']==r['expected_binary_priority'] for r in binary_probes)==bprobe['correct_binary_probes']==14
    assert sum(custom_by_id[r['id']]['priority']==r['expected_binary_priority'] for r in binary_probes)==cprobe['correct_binary_probes']==13
    assert all(r['action']=='REVIEW' for model in [bprobe,cprobe] for r in model['predictions']
        if r['expected_binary_priority'] is not None and r['priority']!=r['expected_binary_priority'])
    probe_table=table(['Số','Nội dung bài đăng','Kỳ vọng','Hai nguồn','Thêm học cặp'],probe_rows,[.7,10,2.1,2.1,2.1],font=8.3)
    for row in probe_table.rows[1:]:
        for cell in row.cells[3:]:
            if 'Sai' in cell.text:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:run.bold=True
    h('Những câu dự đoán sai')
    p('MiniLM hai nguồn sai câu 9 và 11 nên đúng 14/16. MiniLM + học cặp sai thêm câu 3 nên đúng 13/16. Câu 9 chỉ trích dẫn một lỗi đăng nhập; câu 11 vẫn cần hỗ trợ sau khi thử cách khắc phục; câu 3 đã được giải quyết.')
    p('Kết quả đúng/sai được tính theo ngưỡng nhị phân 0,5, kể cả khi ứng dụng yêu cầu giảng viên xem lại. Tất cả các dự đoán sai nêu trên đều có trạng thái REVIEW, không được ứng dụng tự động chấp nhận là quyết định cuối. Bốn câu mơ hồ hoặc ngoài phạm vi không có nhãn kỳ vọng nên không tính vào 16 câu này.')
    # 2
    page('1. Yêu cầu và dữ liệu')
    table(['Yêu cầu học phần','Cách thực hiện'],[
        ['Tối thiểu hai bộ dữ liệu','UPenn và Stanford MOOCPosts, nguồn cố định và hash kiểm chứng'],
        ['Tổng hợp kiến thức đã học','TF-IDF, Logistic Regression, Naive Bayes, SVM, chia tập, cross-validation và đánh giá'],
        ['Phân tích kết quả và lỗi','Precision, Recall, F1, ma trận nhầm lẫn, chuyển nguồn và ví dụ FP/FN'],
        ['Kiến thức bổ sung','Pretrained MiniLM/DeBERTa, fine tuning, calibration, early stopping, augmentation và thử nghiệm nhãn yếu']], [5.5,11.5])
    table(['Nguồn','Raw → sạch','Ưu tiên','Vai trò'],[
        ['UPenn','3.504 → 3.493','298 (8,53%)','Dữ liệu thật'],
        ['Stanford','29.604 → 29.070','5.524 (19,00%)','Dữ liệu thật'],
        ['CustomForum','160 câu tổng hợp','136 nhãn nhị phân','Chưa có nhãn được con người duyệt']], [3,4,4,6])
    p('Hai CSV thật được lấy từ kho nghiên cứu pcla-code/forum-posts-urgency, commit ebf3695f01d2dec197253eb89499f4835bb7d0fa. Nhãn ưu tiên được ánh xạ urgency > 4. Giữ nguyên raw, ghi các hàng bị loại, chuẩn hóa văn bản và loại trùng chính xác; không tự sửa nhãn để tăng điểm.')
    picture(R/'figures/02_labels.png','Hình 1. Phân bố nhãn được tạo từ dữ liệu đã làm sạch.',15)
    p('Ba nguồn nằm trong Project/Datasets/UPenn, Stanford và CustomForum. CustomForum xét nhu cầu hỗ trợ hiện tại, có cặp đang bị chặn / đã giải quyết / cập nhật vẫn bị chặn. Định nghĩa này có thể khác urgency cảm nhận trong benchmark. Câu mơ hồ và ngoài phạm vi không bị ép thành NORMAL.')
    # 3
    page('2. Phương pháp và quy tắc đánh giá')
    table(['Phương pháp','Cách dùng và liên hệ lý thuyết'],[
        ['TF-IDF + LR / SVM / NB','Đặc trưng từ vựng, phân loại tuyến tính hoặc xác suất; tuning trong train'],
        ['MiniLM frozen + LR','Giữ nguyên pretrained encoder, chỉ học LR trên embedding'],
        ['MiniLM fine tuned','Cập nhật encoder và classification head theo nhãn nhiệm vụ'],
        ['Blend','Trộn điểm MiniLM và LR; trọng số, ngưỡng được chọn bằng OOF'],
        ['DeBERTa / MiniLM joint','Học chung hai nguồn, calibration riêng; có nhánh custom tách biệt']], [5,12])
    p('Pretrained là khởi tạo bằng trọng số đã học trước; dự án fine tune cho nhiệm vụ phân loại, không pretrain mô hình ngôn ngữ từ đầu. Pipeline nguồn riêng dùng CV năm fold và ngưỡng OOF. Các nhánh joint dùng fit / calibration / validation / test riêng và giữ lịch sử test.')
    table(['Nguồn joint','Fit','Calibration','Validation','Test'],[
        [source.upper(),*[str(split['source_splits'][source][role]['posts']) for role in ['fit','calibration','validation','test']]] for source in ['upenn','stanford']], [4,3.25,3.25,3.25,3.25])
    p('UPenn chia theo student ID để một sinh viên chỉ thuộc một phần. Stanford chia stratified vì nguồn không cung cấp author/thread ID tương đương. Kiểm tra không có trùng chính xác giữa hai nguồn; chưa loại trừ hết trùng gần nghĩa. Các model nguồn riêng trước đây đã thấy một phần nay dành làm development joint, nên joint khởi tạo lại từ pretrained gốc.')
    p('TF-IDF nhấn mạnh từ đặc trưng và giảm trọng số từ quá phổ biến. LR học trọng số tuyến tính rồi dùng sigmoid cho lớp ưu tiên; SVM học biên phân cách với margin lớn. Naive Bayes kết hợp xác suất từ theo giả định độc lập có điều kiện và smoothing. Regularization và các tham số được chọn trong train; từ điển TF-IDF không fit trên test.')
    h('Chỉ số')
    p('Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN). Accuracy = (TP+TN)/N. Ma trận ghi [TN, FP; FN, TP], hàng là nhãn thật, cột là dự đoán. U = UPenn, S = Stanford, J = học chung U+S; mũi tên chỉ hướng train → test.')
    p('F1 luôn là lớp ưu tiên, không phải Macro-F1. Báo thêm AP, ROC-AUC, P@20 và coverage Review khi có dữ liệu. Accuracy dễ cao do mất cân bằng lớp. Review không bị loại khỏi mẫu số F1; accuracy riêng phần tự động không phải accuracy toàn hệ thống.')
    p('Macro-F1 lấy trung bình F1 hai lớp. AP tóm tắt chất lượng precision–recall theo thứ hạng; ROC-AUC đo phân biệt thứ hạng hai lớp. P@20 là tỷ lệ urgent trong 20 bài điểm cao nhất, không phải Recall trên cả hàng đợi. Các chỉ số này bổ sung cho FP/FN thay vì thay thế việc đọc lỗi.')
    p('Mỗi bảng dùng ngưỡng phát triển đã lưu; ngưỡng 0,5 là chẩn đoán bổ sung khi được ghi rõ. Không tìm ngưỡng bằng test. So sánh pipeline khác thành phần train, calibration hoặc ngân sách tuning không cô lập riêng tác động kiến trúc.')
    # 4
    page('3. Kết quả thực nghiệm với các mô hình cơ sở')
    metrics_table(baseline,'Mac / original')
    p('Bảng 1. Đủ 20 hướng đánh giá của năm pipeline. P, R, F1 và Accuracy dùng toàn bộ test; ma trận cho phép tính lại từng chỉ số. Ngưỡng của SVM là decision margin, không phải xác suất.')
    p('Dummy dự đoán tất cả là NORMAL có F1 ưu tiên bằng 0 dù Accuracy cao. Các mô hình tuyến tính học trực tiếp dấu hiệu từ vựng có thể phù hợp nhiệm vụ urgency. Embedding pretrained học ngữ nghĩa chung không tự bảo đảm giữ các tín hiệu phân loại này; kết quả frozen là bằng chứng của pipeline hiện tại, không kết luận embedding luôn yếu.')
    p('Ví dụ NB train/test UPenn có 61 FP và 30 FN, làm Precision còn 0,3370 và F1 0,4052. LR/SVM giảm được báo nhầm trong cấu hình này; trên Stanford cùng nguồn hai phương pháp có F1 gần nhau, chưa có kiểm định để khẳng định một phương pháp vượt trội. Giả định độc lập từ của NB hoặc mất tín hiệu từ vựng trong embedding là các khả năng giải thích, chưa được thí nghiệm cô lập.')
    p('Chuyển nguồn thường làm giảm điểm: tỷ lệ lớp và cách diễn đạt khác nhau, trong khi một ngưỡng chọn trên nguồn train được giữ khi áp dụng sang nguồn khác. Đây là giải thích phù hợp với số đo; chưa có phân tích nhân quả xác định từng yếu tố.')
    # 5
    page('4. Fine tuning, blending và các chính sách DeBERTa')
    additional=list(fine)+list(blend)+list(large)
    for r in fixed:additional.append({**r,'train_source':'joint','test_source':r['source'],'model':'joint_deberta','threshold':.5,'policy':'fixed 0.5'})
    metrics_table(additional,'Mac / extensions',lambda r:NAMES.get(r['model'],r['model'])+(' / '+r['policy'] if 'policy' in r else ''))
    p('Bảng 2. Fine tuning và blend có bốn hướng; DeBERTa có chính sách nguồn riêng, mixed và chẩn đoán 0,5. Chính sách mixed là lựa chọn chung cho bài mới. Các dòng đổi policy vẫn là một checkpoint, không phải model được train lại.')
    p('Fine tuning tăng F1 so với MiniLM frozen + LR trên cùng nguồn: UPenn 0,5414 → 0,6222, Stanford 0,6210 → 0,8036. Encoder được cập nhật theo nhãn urgency thay vì giữ biểu diễn ngữ nghĩa chung. Tuy nhiên head, augmentation, chọn cấu hình và ngưỡng cũng thay đổi, nên chưa thể quy toàn bộ mức tăng cho việc mở khóa encoder.')
    p('Blend tận dụng hai biểu diễn có thể mắc lỗi khác nhau; mức tăng F1 trên cùng nguồn là quan sát thực nghiệm. Chuyển nguồn không luôn tăng, nên không gọi ensemble là tốt nhất cho mọi dữ liệu. Chi tiết default threshold của các pipeline được giữ trong JSON và bảng CSV đầy đủ.')
    p('Điểm blend = w × điểm MiniLM + (1−w) × điểm LR, với w chọn từ OOF: UPenn 0,75 và Stanford 0,80. Stanford F1 tăng 0,8036 → 0,8092, UPenn 0,6222 → 0,6364; đây là tăng của cả pipeline, chưa chứng minh mức tăng có ý nghĩa thống kê.')
    p('Trên Stanford, blend giảm FP 305 → 293 và FN 158 → 155 so với MiniLM nguồn riêng, giải thích trực tiếp mức tăng F1. Nhưng khi train UPenn rồi test Stanford, blend giảm F1 0,6301 → 0,6268: điểm LR bổ sung có thể giúp lỗi ở nguồn này mà gây hại ở nguồn khác. Trọng số chọn trên OOF nguồn không bảo đảm chuyển nguồn tốt.')
    # 6
    page('5. Những cấu hình đã thử: weighting, augmentation và CV')
    scouts=[];cv=[]
    for source in ['upenn','stanford']:
        selection=load(R/f'fine_tuning/{source}/selection.json')
        for run in selection['scout_runs']:
            scouts.append([source,run['name'],num(run['validation_ap']),num(run['validation_f1_tuned']),run['best_epoch'],'Có' if run['name']=='scout_'+selection['selected'] else ''])
        values=load(R/f'fine_tuning/{source}/cv.json')
        cv.append([source,selection['selected'],num(values['fold_f1_mean']),num(values['fold_f1_std']),values['epochs_for_final']])
    table(['Nguồn','Scout','Val AP','Val F1 tuned','Epoch','Chọn'],scouts,[2,4.6,2.7,2.7,2,3])
    table(['Nguồn','Cấu hình giữ','F1 fold TB','SD','Epoch fit cuối'],cv,[2,5,3.2,3.2,3.6])
    picture(R/'figures/ft_learning_curves.png','Hình 2. Lịch sử loss / validation của các scout nguồn riêng.',16)
    p('Weighted loss tăng đóng góp lớp ưu tiên hiếm; augmentation ở thí nghiệm cũ mở rộng dạng co từ, không tạo thêm nhãn thật độc lập. Early stopping / best checkpoint dựa validation. CV năm fold tạo OOF để chọn ngưỡng và số epoch trước fit cuối. Scout là phép chọn cấu hình, không phải test.')
    p('Ở lượt nguồn riêng, chọn scout theo F1 validation đã tối ưu ngưỡng, hòa thì AP. Vì vậy Stanford chọn plain với F1 0,8021 dù weighted có AP nhỉnh hơn. Quy tắc này khác AP trung bình hai nguồn dùng để chọn checkpoint trong các lượt joint.')
    p('UPenn plain có thể không dự đoán lớp hiếm ở ngưỡng 0,5; weighting giúp trong cấu hình đã đo. Trên nguồn nhiều hơn, weighting/augmentation không mặc nhiên tốt hơn. Giá trị validation là bằng chứng chọn cấu hình, không bảo đảm mọi thay đổi sẽ tổng quát sang test.')
    p('CV năm fold tạo OOF và giảm phụ thuộc một lần chia; early stopping dừng theo validation. Chúng không bổ sung nhãn mới hoặc bảo đảm tăng test F1. Augmentation dạng co từ chưa kiểm tra hết phủ định, trích dẫn hoặc cập nhật mới. Lượt joint tối đa 10 epoch chưa được thử; có thể đặt trần 10 và giữ checkpoint validation tốt nhất, nhưng chưa có số đo để nói sẽ thắng. Đổi trần cũng đổi learning-rate schedule trong mã hiện tại.')
    # 7
    page('6. Phiên Kaggle: baseline từ log thực thi')
    metrics_table(kb,'Kaggle / supplied confusion counts')
    p('Bảng 3. 20 ma trận được chép từ các dòng 1342–1365 của log Kaggle người dùng cung cấp; script summarize_kaggle_baselines.py tính lại P/R/F1/Accuracy từ số đếm. Dấu — nghĩa là không có ngưỡng chi tiết trong trích đoạn này, không phải ngưỡng 0.')
    p('Các dòng log lặp là đầu ra được hiển thị lại, không được đếm thành nhiều lần thử độc lập. Chưa nhập ZIP model hoặc CSV từng bài của phiên Kaggle vào local, nên không khẳng định đã đối chiếu post ID và điểm từng bài Kaggle. AP và AUC của các thí nghiệm khác không được dùng thay cho số đo còn thiếu của phiên Kaggle.')
    # 8
    page('7. Phiên Kaggle: fine tuning và tác động của ngưỡng')
    kr=[]
    for r in kaggle:
        kr.append({**r,'model':'finetuned_minilm','policy':'selected'})
        kr.append({**r['default_threshold_metrics'],'train_source':r['train_source'],'test_source':r['test_source'],'model':'finetuned_minilm','threshold':.5,'policy':'fixed 0.5'})
    metrics_table(kr,'Kaggle / supplied FINAL TEST',lambda r:'MiniLM / '+r['policy'])
    h('Kaggle joint hai nguồn từ log ngày 17 tháng 11')
    metrics_table(joint_kaggle,'Kaggle / supplied joint MiniLM log 2026-11-17',lambda r:'Joint MiniLM / s'+str(r['seed']))
    p('T* là 24 câu kiểm tra tổng hợp, không phải test thật. Phiên joint train hai nguồn thật, không dùng custom; hai seed đều giữ epoch 3 theo AP validation. F1 giảm ở seed 43 vẫn được báo đầy đủ. Các chỉ số P/R/F1/Acc được tính lại từ ma trận trong log, không phải kết quả của một lần huấn luyện mới.')
    picture(R/'figures/kaggle_stanford_confusion.png','Hình 3. Ma trận Stanford của MiniLM nguồn riêng trong log Kaggle cũ, không phải model joint mới.',11.5)
    p('F1 0,6387 là model train UPenn chuyển sang Stanford, không phải Stanford train → Stanford. Cùng checkpoint chuyển nguồn đạt F1 0,6655 tại 0,5; ngưỡng 0,6821 được tối ưu trên nguồn UPenn không bảo đảm tối ưu ở Stanford. Train Stanford → Stanford đạt F1 0,7997 theo ngưỡng đã chọn.')
    p('Log cũ khoảng 61 phút; log joint mới khoảng 16,8 phút, xuất ZIP 182,7 MB, dùng một GPU dù thấy hai. Mô hình học chung hai nguồn trên Kaggle trả lời đúng 12/16 câu tổng hợp bổ sung, so với 14/16 trong thí nghiệm chính; vẫn chưa phân biệt đúng trọn vẹn nhóm nào trong 7 nhóm ba tình huống. Transformers thực tế là 5.20.2 thay vì 4.57.6 yêu cầu: setup chỉ kiểm tra có module nên chưa ép pin khi đã cài. Chưa cô lập nguyên nhân khác biệt; CSV từng bài và weights Kaggle chưa được nhập vào app local.')
    # 9
    page('8. Vì sao mô hình lớn hơn không luôn thắng')
    table(['Đặc điểm','MiniLM nguồn riêng','DeBERTa joint'],[
        ['Giá trị tensor checkpoint',f"{modelsize['models']['MiniLM']['stored_tensor_values']/1e6:.1f} triệu",f"{modelsize['models']['DeBERTa']['stored_tensor_values']/1e6:.1f} triệu"],
        ['Context tối đa','256 token','512 token'],
        ['Training và tuning','Model nguồn riêng; scout + CV','Một model hai nguồn; development riêng'],
        ['Custom dùng train','0','0']], [5,6,6])
    picture(R/'figures/minilm_deberta_fixed_threshold.png','Hình 4. Cùng ngưỡng số 0,5; calibration và dữ liệu fit vẫn khác nhau.',16)
    p('DeBERTa mixed ở ngưỡng 0,07 ưu tiên Recall nên có nhiều FP; giảm F1 không hoàn toàn do encoder. Ở 0,5, DeBERTa đạt F1 0,6119 trên UPenn và 0,7964 trên Stanford. So với MiniLM nguồn riêng ở 0,5, DeBERTa thắng UPenn nhưng chưa thắng Stanford. Vì vậy kết quả không chứng minh model lớn luôn kém.')
    p(f"Loss DeBERTa giảm từ {history[1]['train_loss']:.4f} xuống {history[2]['train_loss']:.4f} giữa epoch 2 và 3, nhưng AP validation trung bình giảm từ {history[1]['mean_validation_ap']:.4f} xuống {history[2]['mean_validation_ap']:.4f}. Checkpoint epoch 2 được giữ. Đây là dấu hiệu tăng epoch không giúp trong lượt này; một run chưa đủ kết luận cơ chế overfitting.")
    p('Ví dụ lỗi nộp bài có cập nhật đã giải quyết vẫn bị cả DeBERTa và MiniLM cũ gán ưu tiên; văn bản dưới giới hạn token nên truncation không giải thích lỗi cụ thể đó. Ít nhãn UPenn, khác định nghĩa nhãn và học shortcut là các giả thuyết cần kiểm tra, không phải nguyên nhân đã chứng minh. Lượt DeBERTa FP32 dừng do áp lực bộ nhớ, chưa có metric chất lượng; lượt BF16 hoàn tất được giữ riêng.')
    # 10
    page('9. So sánh các phương án bổ sung dữ liệu huấn luyện')
    model_legend()
    p('Cả ba phương án đều tinh chỉnh encoder MiniLM và bộ phân loại, không kết hợp TF-IDF. CE là hàm mất mát entropy chéo dùng để học nhãn từng câu; Pair bổ sung yêu cầu câu còn cần hỗ trợ có điểm cao hơn câu đã giải quyết. Seed là số khởi tạo giúp lặp lại thí nghiệm.')
    table(['Nhánh','Dữ liệu và loss','Số run'],[
        ['B real-only','Hai nguồn thật; cân bằng tổng đóng góp theo nguồn','Seed 42, 43'],
        ['C custom CE','80% loss thật + 20% CE trên 82 binary draft train','Seed 42'],
        ['C custom pair','Như CE + 0,25 loss thứ tự bên trong phần custom','Seed 42, 43']], [4,9,4])
    p('CustomForum có 93 train / 36 validation / 31 challenge tổng số; sau bỏ nhãn mơ hồ còn 82 / 30 / 24 binary. Các bộ ba đầy đủ tương ứng 24 / 9 / 7 nhóm, không cắt qua split. Nhãn vẫn là assistant draft; worksheet giữ 0 human approved. Weak supervision được đánh giá bằng các nhánh riêng; nhãn dự thảo không tương đương nhãn được duyệt độc lập.')
    p('Cùng pretrained revision, tokenizer, real fit, batch 64, max length 256, LR 2e-5, tối đa 4 epoch, patience 2. Checkpoint theo AP validation thật trung bình. Calibration Platt chỉ học từ calibration thật riêng; ngưỡng 0,5, Review trong khoảng 0,2–0,8. Không dùng keyword rule ở inference.')
    table(['Seed 42','AP thật TB','Custom val BA','Bộ ba đúng'],[
        [NAMES[c['arm']],num(c['mean_real_validation_ap']),num(c['context_validation']['balanced_accuracy']),str(c['context_validation']['all_three_states_correct'])+' / 9'] for c in [plan['reference'],*plan['candidates']]], [5,4,4,4])
    p('Quy tắc đặt trước: AP thật không giảm quá 0,03 so với MiniLM-B; ưu tiên balanced accuracy trên tập xác thực tổng hợp, hòa thì chọn AP thật cao hơn. MiniLM-C-Pair được chọn để xác nhận với seed 43. Cấu hình được cố định trước khi kiểm tra mô hình này; kết quả MiniLM-B đã được xem trước đó nhưng không dùng để đổi quy tắc. Nhánh được validation đề xuất không đồng nghĩa đã được xác nhận tốt hơn khi dùng thực tế.')
    picture(R/'figures/final_minilm_learning.png','Hình 5. real_only là MiniLM-B; custom_ce là MiniLM-C-CE; custom_pairwise là MiniLM-C-Pair. Hậu tố seed42/43 chỉ số khởi tạo của thí nghiệm.',16)
    # 11
    page('10. Kết quả MiniLM học chung hai nguồn trên dữ liệu thật')
    model_legend()
    realtests=[{**r,'train_source':'joint','model':r['arm']} for r in tests if r['test_source'] in ['upenn','stanford']]
    metrics_table(realtests,'Mac / final joint',lambda r:NAMES[r['arm']]+' / '+str(r['seed']))
    table(['Nhánh','Nguồn','F1 seed 42','F1 seed 43','TB ± SD'],[
        [NAMES[arm],source,num(v['f1']['values_by_seed']['42']),num(v['f1']['values_by_seed']['43']),num(v['f1']['mean'])+' ± '+num(v['f1']['std_across_two_seeds'])] for arm,domains in seeds.items() for source,v in domains.items() if source in ['upenn','stanford']], [4,2.5,3,3,4.5])
    p('Bảng 5. Mỗi nguồn giữ nguyên test và nhãn; mọi bài Review vẫn có trong ma trận. Hai seed mô tả dao động khởi tạo, không phải khoảng tin cậy hoặc kiểm định thống kê.')
    for source in ['upenn','stanford']:
        b=result('real_only',source);c=result(selected,source)
        p(f"{source}: MiniLM-C-Pair đổi F1 {b['f1']:.4f} → {c['f1']:.4f} ở seed 42; FP/FN đổi {b['confusion_matrix'][0][1]}/{b['confusion_matrix'][1][0]} → {c['confusion_matrix'][0][1]}/{c['confusion_matrix'][1][0]}. MiniLM-C-Pair cũng không vượt MiniLM-B ở seed 43. Không điều chỉnh tiếp bằng test trong lượt cuối.")
    table(['Seed 42 / nguồn','Macro-F1','AP','ROC-AUC','P@20'],[
        [NAMES[arm]+' / '+source,*[num(result(arm,source)[k]) for k in ['macro_f1','average_precision','roc_auc','precision_at_20']]] for arm in ['real_only',selected] for source in ['upenn','stanford']], [6,2.75,2.75,2.75,2.75])
    # 12
    page('11. Ma trận nhầm lẫn của các phương án huấn luyện')
    model_legend()
    picture(R/'figures/final_minilm_confusion.png','Hình 6. real_only = MiniLM-B; custom_ce = MiniLM-C-CE; custom_pairwise = MiniLM-C-Pair. Cùng seed 42 và tập kiểm tra; hàng là nhãn thật, cột là nhãn dự đoán.',15)
    p('Các số đếm được đối soát bằng script từ score CSV và nhãn. Lượt cuối không tạo benchmark mới và không bảo đảm chất lượng trên bài ngẫu nhiên ở lĩnh vực khác. Bảng số liệu ở trang trước là dữ liệu nguồn của hình.')
    # 13
    page('12. Ngữ cảnh tổng hợp và câu diễn đạt mới')
    model_legend()
    challenge=[r for r in tests if r['test_source']=='custom_challenge']
    table(['Nhánh / seed','P','R','F1','Acc','Bộ ba đúng'],[[NAMES[r['arm']]+' / '+str(r['seed']),*[num(r[k]) for k in ['precision','recall','f1','accuracy']],str(r['all_three_states_correct'])+' / 7'] for r in challenge],[5,2.2,2.2,2.2,2.2,3.2])
    all_rows.extend({'environment':'Mac / synthetic diagnostic','trial_label':NAMES[r['arm']]+' / '+str(r['seed']),
        'model':r['arm'],'train_source':'joint',**r} for r in challenge)
    table(['Model qua API','Đúng / 16 draft','Lỗi binary draft','Tự xếp 4 câu mơ hồ / ngoài phạm vi'],[
        [{'stanford__finetuned_minilm':'MiniLM tinh chỉnh theo nguồn','joint__reliable_deberta':'DeBERTa học chung hai nguồn','joint__minilm_real_control':'MiniLM-B','joint__minilm_custom_trial':'MiniLM-C-Pair'}.get(r['model_key'],r['model_key']),str(r['correct_binary_probes']),', '.join(r['binary_errors']) or 'Không thấy',str(len(r['unlabelled_probes_automatically_classified']))] for r in probes],[5,2.2,6.2,3.6])
    p('Các bảng trong phần này trình bày thí nghiệm chính; kết quả học chung hai nguồn trên Kaggle được trình bày riêng ở mục 7. Challenge chỉ có 24 binary draft và cùng nguồn soạn với train. Đúng tất cả không chứng minh hiểu mọi ngữ cảnh: trên 16 câu diễn đạt mới, MiniLM-B đúng 14 còn MiniLM-C-Pair đúng 13. Các câu này được lưu trước khi đánh giá MiniLM-C-Pair, không dùng để huấn luyện hoặc chọn mô hình; vẫn là nhãn kỳ vọng do trợ lý soạn, chưa phải gold độc lập. Tách nhóm và không trùng chuỗi chính xác chưa loại trừ mọi mẫu câu gần nhau.')
    cprobe=next(r for r in probes if r['model_key']=='joint__minilm_custom_trial')
    for identity in ['longer_resolution','quoted_problem','still_waiting']:
        row=next(r for r in cprobe['predictions'] if r['id']==identity)
        p(f"MiniLM-C-Pair / {identity}: kỳ vọng draft {row['expected_binary_priority']}, dự đoán {row['priority']}, điểm {row['score']:.4f}. "+row['text'])
    p('Như vậy custom giúp đúng mẫu cập nhật trong bộ tổng hợp, nhưng chưa cải thiện ổn định cách diễn đạt mới hoặc hai benchmark thật. Việc học mẫu câu thay vì khái niệm tổng quát là giả thuyết phù hợp với quan sát, chưa được chứng minh cơ chế. Cần nhãn đích do giảng viên duyệt và bài thật mới để xác nhận.')
    p('Chỉ 82 binary draft train được lặp lại và nhận 20% loss, nên một tập nhỏ có ảnh hưởng lớn dù tổng số bài tăng. Nhãn custom xét nhu cầu hỗ trợ hiện tại, còn dữ liệu thật xét urgency > 4; lệch mục tiêu và học mẫu câu là hai khả năng giải thích. AP xác thực của MiniLM-C-Pair cao hơn không bảo đảm F1 kiểm tra cao hơn: AP đánh giá thứ hạng, F1 phụ thuộc ranh giới quyết định, và validation là mẫu khác test. Chưa có ablation đủ để kết luận riêng từng nguyên nhân.')
    p(f"Kiểm tra AI ngày 17/11: đã đọc {ai_audit['rows_semantically_audited']} câu, giữ 136 nhãn binary, 12 REVIEW và 12 ngoài phạm vi theo rubric hỗ trợ hiện tại. Đây không phải human approval; worksheet giữ 0 hàng duyệt. Cả 40 cập nhật still-blocked dùng cùng một hậu tố, cho thấy độ đa dạng còn hẹp. Nhãn có vẻ nhất quán không chứng minh tăng dữ liệu này sẽ tăng chất lượng thực tế. Audit lưu riêng và tái tạo bằng audit_custom_labels_ai.py.")
    # 14
    page('13. Cùng MiniLM vì sao kết quả khác nhau')
    p('Hai notebook dùng cùng encoder pretrained all-MiniLM-L6-v2 ở cùng revision, nhưng không phải cùng thí nghiệm hoặc cùng model đã train. Notebook urgency_minilm_training.ipynb tạo model nguồn riêng; final_best_model.ipynb tạo MiniLM-B, tinh chỉnh all-MiniLM-L6-v2 bằng cả UPenn và Stanford, không kết hợp TF-IDF. Model mới khởi tạo lại từ pretrained gốc. Bảng dưới so sánh hai phương án trên cùng tập kiểm tra; kết quả Kaggle được trình bày riêng ở mục 7.')
    table(['Yếu tố','MiniLM nguồn riêng trước đây','MiniLM hai nguồn'],[
        ['Dữ liệu gradient','UPenn 2.802 bài thật; Stanford 23.256 bài thật, mỗi nguồn một model','UPenn 2.257 + Stanford 18.604 bài, một model; phần còn lại dành cho calibration / validation'],
        ['Cấu hình / trọng số','UPenn weighted + augmentation; Stanford plain','Cân bằng tổng loss 50/50 theo nguồn, giữ tỷ lệ lớp tự nhiên'],
        ['Phát triển cấu hình','Ba scout và CV năm fold mỗi nguồn; fit cuối 3 epoch','Hai seed; tối đa 4 epoch, patience 2; checkpoint theo AP validation thật trung bình'],
        ['Ngưỡng và calibration','Ngưỡng theo F1 OOF từng nguồn; điểm head gốc','Platt trên calibration thật riêng; ngưỡng chung 0,5, khoảng Review 0,2–0,8']], [3.3,6.85,6.85],font=8.7)
    table(['Test','Pipeline','P','R','F1','FP / FN'],[
        [source,'Nguồn riêng',num(old['precision']),num(old['recall']),num(old['f1']),f"{old['confusion_matrix'][0][1]} / {old['confusion_matrix'][1][0]}"]
        if kind=='old' else [source,'MiniLM-B seed 42',num(new['precision']),num(new['recall']),num(new['f1']),f"{new['confusion_matrix'][0][1]} / {new['confusion_matrix'][1][0]}"]
        for source in ['upenn','stanford']
        for old,new in [(next(r for r in fine if r['train_source']==r['test_source']==source),result('real_only',source))]
        for kind in ['old','new']], [2.3,4.5,2.3,2.3,2.3,3.3])
    h('Nguyên nhân trực tiếp của chênh lệch điểm')
    p('UPenn: MiniLM-B giữ TP = 42 và FN = 19 nhưng giảm FP 32 → 23, nên F1 tăng 0,6222 → 0,6667. Stanford: MiniLM-B giảm FP 305 → 168, tăng Precision 0,7564 → 0,8272, nhưng FN tăng 158 → 301 và Recall giảm 0,8570 → 0,7276. Số bỏ sót tăng làm F1 giảm 0,8036 → 0,7742. Đây là giải thích từ số đếm, không phải suy đoán cơ chế bên trong encoder.')
    h('Các yếu tố có thể tạo ra thay đổi đó')
    p('MiniLM-B dùng ít bài Stanford trực tiếp để cập nhật trọng số hơn, chia đều đóng góp hai nguồn thay vì chuyên biệt Stanford, chọn checkpoint theo AP chung và dùng calibration / ranh giới chung. Các thay đổi này có thể tạo đánh đổi giữa hai nguồn và giữa Precision–Recall. Nhiều yếu tố thay cùng lúc nên chưa xác định yếu tố nào gây ra bao nhiêu mức giảm; không thể kết luận chỉ vì học chung hoặc vì calibration.')
    previous_stanford=next(r for r in fine if r['train_source']==r['test_source']=='stanford')
    p(f"AP Stanford cũng giảm {previous_stanford['average_precision']:.4f} → {result('real_only','stanford')['average_precision']:.4f}; AP không phụ thuộc một ngưỡng binary, nên chất lượng thứ hạng cũng thay đổi. MiniLM cũ ở raw threshold 0,5 đạt F1 0,8038, còn MiniLM-B sau hiệu chỉnh ở ngưỡng 0,5 là 0,7742; cùng số 0,5 không có cùng ý nghĩa khi thang điểm khác. Tên final_best_model là notebook prototype được chọn cho mục tiêu chung; nếu chỉ xét F1 Stanford trong các run đã đo, blend 0,8092 cao hơn. Chênh một câu trên 16 probe draft chưa chứng minh ưu thế thực tế của MiniLM-B.")
    # 15
    page('14. Phân tích lỗi thật và chi phí Review')
    p('Các ví dụ sau thuộc MiniLM-B, seed 42: all-MiniLM-L6-v2 được tinh chỉnh trên UPenn và Stanford, không dùng dữ liệu tổng hợp hoặc TF-IDF. Lỗi của nhánh custom và các model cũ được lưu riêng cùng score CSV; không trộn điểm giữa các model. Ký tự điều khiển không in được được biểu diễn [U+XXXX]; dữ liệu CSV gốc giữ nguyên.')
    for source in ['upenn','stanford']:
        for kind in ['FP','FN']:
            r=errors[control][source][kind][0]
            h(source.upper()+' / '+kind+' / '+r['post_id'])
            p(f"Nhãn {r['label']}; dự đoán {r['predicted']}; điểm {float(r['score']):.4f}; action {r['action']}. "+r['text'][:210]+('…' if len(r['text'])>210 else ''))
    p('Chọn FP có điểm cao nhất và FN có điểm thấp nhất theo quy tắc cố định; đây là lỗi tự tin cao, không phải mẫu ngẫu nhiên đại diện. Nhãn benchmark phản ánh urgency; ngữ cảnh khóa học, phủ định và trích dẫn có thể khác nhu cầu hỗ trợ hiện tại. Không tự kết luận nhãn gốc sai từ một trích đoạn.')
    model_legend(include_ce=False)
    table(['Seed 42 / nguồn','Coverage tự động','Review','Lỗi tự động','Urgent bỏ sót tự động'],[
        [NAMES[arm]+' / '+source,num(result(arm,source)['automatic_coverage']),result(arm,source)['review_count'],result(arm,source)['automatic_errors'],result(arm,source)['urgent_automatically_missed']] for arm in ['real_only',selected] for source in ['upenn','stanford']], [5.5,3,2,2.5,4])
    p('Review dùng khoảng điểm, không phải bộ phát hiện ngoài phạm vi. MiniLM-C-Pair vẫn tự gán nhãn cho 2/4 câu mơ hồ hoặc ngoài phạm vi. Không tính Review là đúng sẵn hoặc bỏ chúng khỏi F1. Điểm cao hay đã calibration vẫn có thể sai.')
    # 16
    page('15. Hướng dẫn web phân loại một bài đăng')
    p('Khởi động bằng lệnh bash Project/scripts/launchers/start_app.sh, giữ cửa sổ đó mở, rồi truy cập http://127.0.0.1:8765. Web gợi ý bài cần phản hồi trước; web không soạn câu trả lời cho câu hỏi của sinh viên.')
    table(['Bước','Thao tác'],[
        ['1. Chọn nội dung','Vào Phân loại bài đăng, nhập hoặc dán một bài học tập bằng tiếng Anh. Nêu rõ vấn đề hiện tại và việc đã giải quyết hay chưa.'],
        ['2. Chọn mô hình','Mặc định là MiniLM-B, seed 42: MiniLM được tinh chỉnh trên UPenn và Stanford, không kết hợp TF-IDF. Danh sách còn các model cũ và bản CustomForum thử nghiệm để đối chiếu.'],
        ['3. Xem kết quả','Bấm Phân loại bài đăng. Đọc nhãn và nội dung gốc trước khi quyết định; điểm cao không bảo đảm đúng.']], [4,13],font=9.4)
    picture(R/'screenshots/12_final_minilm_control.png','Hình 7. Ví dụ chạy thật: bài đã báo giải quyết vẫn có gợi ý binary sai; thẻ Review yêu cầu người đọc kiểm tra.',16)
    table(['Kết quả hiển thị','Người dùng nên làm gì'],[
        ['Cần ưu tiên phản hồi','Xem và xử lý sớm; vẫn kiểm tra nội dung, mức khẩn cấp và ngữ cảnh.'],
        ['Chưa cần ưu tiên','Xếp sau nhóm cần hỗ trợ sớm; không hiểu là bài chắc chắn không cần trả lời.'],
        ['Cần giảng viên xem lại','Đọc thêm ngữ cảnh hoặc hỏi rõ. Gợi ý binary đang chờ xem lại chưa phải quyết định tự động.']], [5.5,11.5],font=9.4)
    p('Nếu đổi nội dung hoặc đổi model, bấm Phân loại bài đăng lại để có kết quả mới. Điểm của các model khác nhau không nên so trực tiếp; mỗi model dùng ngưỡng riêng. Bài rất dài có thể bị cắt khi phân tích, nên giảng viên cần đọc đầy đủ.')
    # 17
    page('16. Hướng dẫn web danh sách và các trường hợp khác')
    p('Để xử lý nhiều bài, chọn model ở màn Phân loại, rồi mở Hàng đợi phản hồi. Bấm Dùng danh sách mẫu để thử hoặc Nhập CSV để dùng dữ liệu của mình. CSV cần cột text, post_text hoặc text_clean; tối đa 500 bài mỗi lần. Ví dụ tối thiểu: dòng đầu text, dòng sau là nội dung bài đăng.')
    picture(R/'screenshots/04_queue.png','Hình 8. Danh sách được xếp theo điểm giảm dần; nút Xuất kết quả tải CSV có nhãn, điểm và action Review.',11)
    p('Đọc các bài đầu danh sách và những bài có thẻ xem lại. Bấm Xuất kết quả để tải CSV. Nếu đổi model, nhập / phân loại lại danh sách; việc đổi lựa chọn không tự tính lại kết quả cũ.')
    p('Để xem số liệu, mở Kết quả thực nghiệm và chọn Hướng đánh giá. Bảng so sánh các model trên tập test; ma trận bên dưới thuộc model có F1 cao nhất trong hướng đang chọn. Đây là kết quả đánh giá đã lưu, không phải độ đúng của câu vừa nhập.')
    table(['Trường hợp','Cách xử lý'],[
        ['Ô nhập rỗng hoặc báo sai nội dung','Nhập bài tiếng Anh có chữ cái; mỗi bài tối đa 12.000 ký tự, rồi thử lại.'],
        ['CSV không được nhận','Kiểm tra tên cột và số dòng; chia file lớn thành các phần không quá 500 bài. CSV có dấu phẩy trong nội dung cần đặt trường văn bản trong dấu nháy kép.'],
        ['Tiếng Việt hoặc câu ngoài học tập','Không dùng nhãn đó làm kết luận. Phạm vi kiểm tra là bài diễn đàn học tập tiếng Anh; web chưa có bộ phát hiện ngoài phạm vi đã kiểm định.'],
        ['Câu mơ hồ, mâu thuẫn hoặc dự đoán sai','Giảng viên đọc toàn bộ luồng, hỏi thêm rồi tự quyết định bài cần phản hồi trước. Nhãn chưa cần ưu tiên không ngăn người dùng phản hồi ngay.'],
        ['Không mở được web hoặc không có kết quả','Kiểm tra lệnh khởi động còn chạy, dùng đúng địa chỉ ở trên và đọc thông báo lỗi. Khi cần, dừng bằng Ctrl+C, chạy lại lệnh và tải lại trang.']], [5.1,11.9],font=9)
    # 18
    page('17. Tái tạo, quyết định cuối và tài liệu tham khảo')
    table(['Mục đích','Lệnh / tệp từ thư mục machine'],[
        ['Chạy ứng dụng','bash Project/scripts/launchers/start_app.sh'],
        ['Train model cuối, giữ run đã hoàn tất','bash Project/scripts/launchers/reproduce.sh --best-minilm'],
        ['Tái chạy so sánh cả nhánh custom','bash Project/scripts/launchers/reproduce.sh --final-minilm'],
        ['Sinh lại hình, ảnh và báo cáo','bash Project/scripts/launchers/reproduce.sh'],
        ['Kiểm tra tính toàn vẹn','python3 Project/scripts/verify_evidence.py'],
        ['Notebook chính: Internet ON, T4 ×2','Project/kaggle/final_best_model.ipynb → Run All'],
        ['Chạy từng phương pháp trong báo cáo','Project/kaggle/teacher_experiments/README.md; chọn notebook 00–10 trong cùng thư mục'],
        ['Đối chiếu ba nguồn và các nhánh thất bại','Project/kaggle/final_minilm_trial.ipynb'],
        ['Mô hình đã chọn và kết quả','Project/FINAL/FINAL_MODEL; Project/reports/final_minilm_trial'],
        ['Bảng mọi số đo và nguồn','Project/reports/all_trials_comparison.csv; all_trials_comparison.json']], [5.7,11.3])
    checked=load(F/'best_notebook_validation.json')
    p('Minh chứng web được chụp bằng Chrome chạy thật: đã đối soát 15 model qua API, kiểm tra nhập câu, nhãn / điểm, hàng đợi, xuất CSV và màn hình mobile. Các nút ví dụ chỉ điền văn bản, kết quả vẫn do model tính.')
    p(f"Notebook tự chứa mã và tải dữ liệu, không cần đính kèm dataset/API key. Bản chính đã kiểm tra local {checked['executed_code_cells']} code cell bằng training đã lưu. Bộ 00–10 tách từng phương pháp; upload một file theo README. Bản 08 chạy model hai nguồn đã chọn, 09 so sánh ba nguồn; 00 chuẩn bị dữ liệu, 10 phát lại lịch sử, không train. Các notebook training mặc định huấn luyện mới. Local QA không phải Kaggle GPU; log Kaggle đo thực tế giữ riêng ở mục 7. Giới hạn 10 epoch chưa được thử.")
    p('Ứng dụng dùng MiniLM-B, seed 42: all-MiniLM-L6-v2 được tinh chỉnh trên hai nguồn UPenn và Stanford, không kết hợp TF-IDF; notebook chính chỉ train hai nguồn thật và báo cả hai seed. Quyết định này dựa trên tổng đánh giá sau thử nghiệm, không phải tuyên bố test chưa từng được xem. MiniLM-C-Pair đạt điều kiện lựa chọn trên tập xác thực nhưng chưa tốt hơn MiniLM-B khi đánh giá trên tập kiểm tra. Trong các thí nghiệm chính, phương án kết hợp MiniLM và TF-IDF có F1 Stanford cao hơn; mô hình chỉ dùng dữ liệu thật trả lời đúng nhiều câu kiểm tra bổ sung hơn mô hình thêm dữ liệu tổng hợp. Chưa có nghiên cứu người dùng hoặc bộ đích mới để chứng minh ưu thế thực tế. Không tuning thêm theo test sau lượt cuối.')
    h('Tài liệu và dữ liệu đối chiếu')
    for text in [
        'Švábenský và cộng sự (2023). Towards Generalizable Detection of Urgency of Discussion Forum Posts. arxiv.org/abs/2307.07614.',
        'Dữ liệu và mã nghiên cứu: github.com/pcla-code/forum-posts-urgency, commit cố định nêu ở trang dữ liệu; Stanford MOOC Posts của Agrawal và Paepcke.',
        'Model cards: huggingface.co/sentence-transformers/all-MiniLM-L6-v2; huggingface.co/microsoft/deberta-v3-base.',
        'scikit-learn: Model evaluation, Calibration, Common pitfalls. Transformers: Training. Tài liệu môn học trong machine: Prob-model, SVM và Model-assessment.']:
        paragraph=p(text)
        for run in paragraph.runs:run.font.size=Pt(8)
    h('Giới hạn')
    p('Hai nguồn tiếng Anh; nhãn chủ quan; không có thread/author ID Stanford; chưa kiểm tra đầy đủ near-duplicate; test lịch sử đã được xem. Dữ liệu custom/probe cùng nguồn soạn và chưa có con người duyệt. Không suy ra độ đúng cho mọi câu tùy ý hoặc cho tiếng Việt. Kết luận nguyên nhân được ghi là giả thuyết nếu chưa có thí nghiệm cô lập.')
    # Full machine-readable records retain all auxiliary statistics, not only printed columns.
    (R/'all_trials_comparison.json').write_text(json.dumps(all_rows,indent=2,ensure_ascii=False)+'\n')
    columns=['environment','trial_label','train_source','test_source','model','seed','policy','threshold','precision','recall','f1','accuracy','macro_f1','average_precision','roc_auc','precision_at_20','recall_at_20','brier_score','automatic_coverage','review_count','confusion_matrix']
    with (R/'all_trials_comparison.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for record in all_rows:writer.writerow({**record,'confusion_matrix':cm(record)})
    canonical_save(d,R/'bao_cao_do_an_hoc_may.docx')
    (R/'official_report_plan.json').write_text(json.dumps({'planned_pages':len(pages),'page_titles':pages,'printed_comparison_rows':len(all_rows),'status':'render and visual verification required'},indent=2,ensure_ascii=False)+'\n')
    print('Official report:',len(pages),'planned pages;',len(all_rows),'comparison rows.')
if __name__=='__main__':main()
