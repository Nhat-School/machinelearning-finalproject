"""Measured optimization using only already-trained models, without replacing prior results."""
import json
from pathlib import Path
from docx.shared import Cm,Pt
from build_report import table,figure
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports'
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def append_blend(document):
    if not (R/'ensembles/results.json').exists():return
    fine=load(R/'fine_tuning/results.json');blend=load(R/'ensembles/results.json')
    def p(text):document.add_paragraph(text)
    def page(title):document.add_page_break();document.add_heading(title,1)
    def fmt(x):return f'{x:.4f}'
    page('Tối ưu bằng cách trộn hai mô hình đã huấn luyện')
    p('Thí nghiệm này dùng lại điểm MiniLM fine-tuned và TF-IDF Logistic Regression của phiên Mac; không train lại encoder. '
      'Điểm trộn = w × điểm MiniLM + (1 − w) × điểm LR. Duyệt 21 giá trị w từ 0 đến 1 với bước 0,05; '
      'tại mỗi w chọn ngưỡng tối đa F1 OOF train nguồn. Giữ nguyên post ID và kiểm tra nhãn hai bảng khớp nhau. '
      'Cấu hình được lưu trước khi đọc điểm test. Ngưỡng của model MiniLM thuần đã tối ưu OOF nên không thử đổi tùy ý trên test.')
    selections={s:load(R/f'ensembles/{s}/selection.json') for s in ['upenn','stanford']}
    table(document,['Nguồn','w MiniLM','Ngưỡng trộn','OOF MiniLM','OOF trộn'],
        [[s,fmt(d['selected']['minilm_weight']),fmt(d['selected']['threshold']),
          fmt(next(c['oof_f1'] for c in d['candidates'] if c['minilm_weight']==1)),fmt(d['selected']['oof_f1'])]
         for s,d in selections.items()],[4.2,3.2,3.2,3.2,3.2])
    figure(document,'blend_oof_search.png','Hình 23. Tìm trọng số trên OOF nguồn; không tìm cấu hình bằng test. CV này vẫn là development diagnostics.')
    table(document,['Train → test','F1 MiniLM','P trộn','R trộn','F1 trộn'],
        [[r['train_source']+' → '+r['test_source'],fmt(next(f['f1'] for f in fine if f['train_source']==r['train_source'] and f['test_source']==r['test_source'])),
          fmt(r['precision']),fmt(r['recall']),fmt(r['f1'])] for r in blend],[6.2,2.7,2.7,2.7,2.7])
    p('Stanford → Stanford tăng từ 0,803564 lên 0,809199; UPenn → UPenn tăng từ 0,622222 lên 0,636364. '
      'UPenn → Stanford giảm từ 0,630079 xuống 0,626834, nên không kết luận trộn luôn có lợi. '
      'Đây là điểm của phiên Mac, không thay thế F1 0,799659 đã đo trên Kaggle và không bảo đảm phiên Kaggle trộn cũng tăng.')
    p('Trọng số được chọn sau khi base models đã trải qua tuning và early stopping; các OOF của LR và MiniLM còn dùng số fold khác nhau. '
      'Đây là chọn cấu hình ở giai đoạn phát triển, không phải nested CV không thiên lệch. Các test đã được xem từ trước; '
      'so sánh là exploratory. Chưa có kết luận về ý nghĩa thống kê hoặc F1 tối ưu toàn cục.')
    page('Dùng model trộn và tối ưu nhanh trên kết quả Kaggle cũ')
    screenshot=R/'screenshots/08_hybrid.png'
    if screenshot.exists():
        para=document.add_paragraph();para.paragraph_format.keep_with_next=True;para.add_run().add_picture(str(screenshot),width=Cm(17))
        para=document.add_paragraph('Hình 24. App chạy thật với Stanford MiniLM + TF-IDF blend; đối chiếu API và giao diện trong browser_checks.json.')
        for run in para.runs:run.italic=True;run.font.size=Pt(8.5)
    p('App có thêm lựa chọn MiniLM + TF-IDF blend; các lựa chọn cũ vẫn có để đối chiếu. '
      'models/urgency_hybrid_stanford chứa đủ MiniLM weights, tokenizer, tfidf_logreg.joblib, decision_config.json và predict.py. '
      'Đã kiểm tra prediction CPU offline. Ngưỡng Stanford 0,393199 thuộc điểm trộn, không áp dụng cho điểm MiniLM thuần.')
    p('Notebook ngắn kaggle/optimize_existing_models.ipynb không train lại encoder. Nếu session cũ còn các tệp Project thì Run All trực tiếp. '
      'Trong session mới, thêm output của notebook cũ bằng Add Input; notebook tự tìm OOF, test predictions và model. '
      'ZIP cũ 171,7 MB chỉ có fine-tuned model và fine-tuning statistics, thiếu OOF của LR nên riêng ZIP đó chưa đủ cho phép trộn.')
    p('Notebook huấn luyện đầy đủ hiện cũng có bước trộn và hiển thị tổng 28 bộ kết quả local gồm 20 baseline/frozen, bốn fine-tuned và bốn hybrid. '
      'Muốn kiểm tra lại trên máy: chạy bash Project/scripts/launchers/reproduce.sh. Muốn dùng app: bash Project/scripts/launchers/start_app.sh rồi chọn Stanford MiniLM + TF-IDF blend. '
      'Model trộn cần suy luận cả MiniLM và LR; đây là một ensemble sử dụng hai thành phần, không phải chỉ điều chỉnh một ngưỡng của MiniLM.')
