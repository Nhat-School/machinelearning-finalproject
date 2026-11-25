"""Explain larger-model tradeoffs and a prospective three-dataset MiniLM experiment."""
import json
from pathlib import Path
from build_report import table,figure
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports'
def append_model_comparison(d):
    path=R/'reliability/model_comparison_discussion.json'
    if not path.exists():return
    data=json.loads(path.read_text());comparisons=data['comparisons'];models=data['models']
    def p(text):d.add_paragraph(text)
    def page(title):d.add_page_break();d.add_heading(title,1)
    def f(number):return f'{number:.4f}'
    page('Vì sao mô hình lớn hơn không luôn thắng')
    p('Kết quả cho thấy tăng kích thước encoder không tự bảo đảm phân loại tốt hơn cho mọi nguồn và mọi tiêu chí. '
      'Tuy nhiên, thí nghiệm này cũng không chứng minh MiniLM luôn tốt hơn DeBERTa. Cần xét cách huấn luyện, '
      'định nghĩa nhãn, cách hiệu chỉnh điểm và ngưỡng quyết định trước khi so sánh năng lực mô hình.')
    table(d,['Đặc điểm','MiniLM fine tuned','DeBERTa joint'],[
        ['Kích thước checkpoint',f"Khoảng {models['MiniLM']['stored_tensor_values']/1e6:.1f} triệu giá trị tensor",f"Khoảng {models['DeBERTa']['stored_tensor_values']/1e6:.1f} triệu giá trị tensor"],
        ['Giới hạn token',models['MiniLM']['max_length'],models['DeBERTa']['max_length']],
        ['Dữ liệu fit','Mỗi nguồn có một model riêng','Một model học chung hai nguồn'],
        ['Phát triển cấu hình','Ba scout và CV năm fold','Một cấu hình joint và validation riêng'],
        ['Nhãn custom dùng trong training','0',str(data['actual_deberta_custom_training_rows'])]], [4.2,6.4,6.4])
    p('Số phần tử tensor được đọc trực tiếp từ header của checkpoint đã lưu; có thể gồm buffer và được dùng như kích thước xấp xỉ. '
      'Đặc tả pretrained được đối chiếu với model card của Sentence Transformers và Microsoft. '
      'Hai pipeline có mức tuning và thành phần dữ liệu khác nhau, nên đây là so sánh thực nghiệm của sản phẩm, chưa cô lập tác động riêng của số tham số.')
    table(d,['Test','Model','Ngưỡng đã chọn','Precision','Recall','F1'],[[r['source'],r['model'],f(r['threshold']),
        f(r['precision']),f(r['recall']),f(r['f1'])] for r in comparisons if r['policy']=='published_selected'],[2.3,3.2,3.1,2.8,2.8,2.8])
    p('Bảng trên dùng ngưỡng từng pipeline đã chọn bằng dữ liệu phát triển: MiniLM theo OOF F1 nguồn, DeBERTa theo balanced accuracy validation. '
      'DeBERTa có Recall cao hơn nhưng nhiều báo nhầm, làm Precision và F1 giảm. Khác biệt ngưỡng là một phần nguyên nhân quan sát được; '
      'không thể quy toàn bộ chênh lệch F1 cho encoder lớn hơn.')
    d.add_heading('Đối chiếu thêm tại ngưỡng cố định',2)
    table(d,['Test','Model','P tại 0,5','R tại 0,5','F1 tại 0,5','FP / FN'],[[r['source'],r['model'],f(r['precision']),
        f(r['recall']),f(r['f1']),str(r['false_positives'])+' / '+str(r['false_negatives'])] for r in comparisons if r['policy']=='fixed_0_5'],[2.3,3.2,2.8,2.8,2.8,3.1])
    p('Ở cùng ngưỡng số 0,5, DeBERTa có F1 cao hơn trên UPenn; MiniLM có F1 cao hơn trên Stanford. '
      'Điểm DeBERTa đã được Platt calibration còn MiniLM dùng điểm head gốc, nên bảng này vẫn chưa đồng nhất mọi điều kiện. '
      'Ngưỡng 0,5 là phép chẩn đoán, không được chọn bằng cách tìm điểm test tốt nhất và không thay chính sách Review của app.')
    page('Năng lực biểu diễn cần phù hợp với dữ liệu và mục tiêu')
    figure(d,'minilm_deberta_fixed_threshold.png','Hình 32. Số đo thật tại ngưỡng cố định 0,5 trên cùng post ID test; calibration và thành phần train vẫn khác nhau.')
    d.add_heading('Những điều thí nghiệm trực tiếp cho thấy',2)
    pair={r['case_type']:r for r in data['context_examples']}
    blocked,resolved=pair['blocked'],pair['resolved']
    p('Một tình huống tổng hợp mô tả lỗi nộp bài, rồi thêm cập nhật hệ thống đã hoạt động và không cần hỗ trợ. '
      f"DeBERTa đổi điểm từ {float(blocked['deberta_score']):.4f} thành {float(resolved['deberta_score']):.4f}, vẫn PRIORITY; "
      f"MiniLM đổi từ {float(blocked['minilm_score']):.4f} thành {float(resolved['minilm_score']):.4f}, cũng vẫn PRIORITY. "
      'Đây là lỗi theo nhãn dự thảo của tình huống, không phải accuracy trên một benchmark độc lập đã được giảng viên xác nhận.')
    p(f"Bản có cập nhật dài {resolved['deberta_tokens']} token DeBERTa và {resolved['minilm_tokens']} token MiniLM, đều dưới giới hạn đầu vào tương ứng. "
      'Do đó việc cắt văn bản không giải thích được lỗi cụ thể này. Kích thước context lớn hơn chưa đủ để bảo đảm mô hình sử dụng đúng thông tin trạng thái mới.')
    history=data['deberta_history'];second,third=history[1],history[2]
    p(f"Loss DeBERTa tiếp tục giảm từ {second['train_loss']:.4f} ở epoch 2 xuống {third['train_loss']:.4f} ở epoch 3, "
      f"nhưng AP validation trung bình giảm từ {second['mean_validation_ap']:.4f} xuống {third['mean_validation_ap']:.4f}. "
      'Đây là dấu hiệu cho thấy chỉ tăng epoch chưa chắc có ích trong cấu hình hiện tại. Một lượt chạy chưa đủ chứng minh nguyên nhân là overfitting; '
      'checkpoint epoch 2 đã được giữ theo tiêu chí validation đặt trước.')
    d.add_heading('Các giả thuyết cần kiểm tra riêng',2)
    positive=data['data_plan']['source_splits']['upenn']['fit']['urgent']
    p(f"Phần fit UPenn chỉ có {positive} bài ưu tiên. Thông tin cho lớp hiếm còn hạn chế dù encoder có nhiều tham số. "
      'Mô hình có thể học các liên hệ giữa cách diễn đạt vấn đề và nhãn mà chưa xử lý vững phủ định, trích dẫn hoặc cập nhật đã giải quyết. '
      'Ví dụ sai phù hợp với giả thuyết này, nhưng chưa phải bằng chứng cơ chế nội bộ hoặc bằng chứng nhãn benchmark bị sai.')
    p('Nhãn hai nguồn thật là urgency > 4, trong khi ứng dụng còn quan tâm bài hiện có cần giảng viên hành động hay không. '
      'Trạng thái luồng hoặc việc đã có người trả lời có thể không có trong văn bản đầu vào. Cần thống nhất định nghĩa nhãn trước khi bổ sung dữ liệu; '
      'không tự sửa nhãn gốc để làm chỉ số tăng. Model pretrained cung cấp biểu diễn ngôn ngữ, còn phân loại theo tiêu chí dự án phải được học và kiểm tra trên dữ liệu phù hợp.')
    p('Vì vậy kết luận phù hợp là pipeline lớn hơn chưa thắng ổn định trong thí nghiệm này. Để giải thích chính xác hơn, '
      'cần giữ cách chia dữ liệu, tiêu chí đánh giá và ngân sách tuning nhất quán, rồi thay từng yếu tố. '
      'Ưu tiên tiếp theo là kiểm tra tác động của dữ liệu ngữ cảnh đã duyệt, cùng chi phí báo nhầm, bỏ sót và bài cần Review.')
    page('Có thể huấn luyện MiniLM trên ba bộ dữ liệu')
    custom=data['custom_dataset'];splits=custom['splits']
    p('Có thể fine tune toàn bộ encoder MiniLM và classification head với UPenn, Stanford và CustomForum. '
      'DeBERTa ở phần trên chỉ học hai nguồn thật; lỗi khi khảo sát câu custom không phải kết quả một model DeBERTa đã train ba nguồn. '
      'Phần thử nghiệm MiniLM cuối phía sau tách riêng đối chứng hai nguồn và các nhánh dùng nhãn custom tổng hợp như weak supervision.')
    table(d,['Nguồn','Vai trò trong thí nghiệm đề xuất','Quy tắc sử dụng'],[
        ['UPenn và Stanford','Dữ liệu thật làm nền','Giữ test lịch sử và nhãn gốc; dùng fit đã lưu'],
        ['CustomForum train',f"Có {splits['train']} hàng, trong đó 82 binary draft",'Tách nhãn human approved và thí nghiệm weak supervision'],
        ['CustomForum validation và challenge',f"{splits['validation']} validation và {splits['challenge']} challenge",'Tách theo nhóm tình huống; không cập nhật trọng số bằng các hàng này']], [3.5,6.2,7.3])
    p(f"Tổng bộ custom có {custom['rows']} câu nháp và hiện {custom['human_approved_rows']} hàng được duyệt. "
      'Các nhãn REVIEW và OUT_OF_SCOPE không được ép thành NORMAL để đưa vào bài toán binary. '
      'Người duyệt cần xác nhận cả ví dụ đã giải quyết lẫn cập nhật vẫn bị chặn, nhằm hạn chế việc học shortcut dựa trên từ Update.')
    d.add_heading('Cách so sánh để biết bộ thứ ba có giúp hay không',2)
    table(d,['Nhánh','Dữ liệu và khởi tạo','Trạng thái'],[
        ['A tham chiếu','Hai MiniLM nguồn riêng hiện có','Đã chạy; số liệu ở phần trên'],
        ['B đối chứng','Pretrained MiniLM rồi fit chung UPenn và Stanford','Xem thử nghiệm cuối'],
        ['C thêm custom','Cùng cấu hình B, thêm custom train; ghi rõ nguồn nhãn','Xem thử nghiệm cuối']], [3.2,10,3.8])
    p('So C với B để kiểm tra đóng góp của dữ liệu thứ ba; giữ model, split, seed, tokenizer và quy tắc chọn checkpoint nhất quán. '
      'Tỷ lệ lấy mẫu hoặc trọng số custom cần chọn trên dữ liệu phát triển: dữ liệu Stanford nhiều hơn đáng kể, còn lặp quá nhiều một ít câu tổng hợp cũng có thể gây overfitting. '
      'Đánh giá riêng hai test thật, tập tình huống có nhãn đã duyệt, báo nhầm, bỏ sót và tỷ lệ Review. Cần thêm bài thật mới để xác nhận ngoài tập đã xem.')
    p('Đề xuất B và C bắt đầu từ cùng pretrained MiniLM, giữ các folder model cũ. Nếu tiếp tục đúng checkpoint Stanford đã fine tune, '
      'phải lưu ý checkpoint đó từng học toàn bộ train Stanford cũ, gồm những hàng nay được dành cho calibration và validation của thí nghiệm joint. '
      'Không thể gọi các hàng đó là chưa được model thấy; cần development data mới hoặc thiết kế lại phép kiểm tra. Dùng lại kiến trúc MiniLM là lựa chọn đơn giản hơn để giữ phép so sánh sạch.')
    p('finetune_minilm.py và notebook MiniLM cũ vẫn là thí nghiệm model nguồn riêng. Nhánh joint có trainer và notebook cuối riêng, '
      'ghi lại checkpoint, seed và nhãn custom đã dùng. Xem phần kết quả cuối để biết số đo thực tế; '
      'thêm một CSV vào folder của notebook cũ không tự thay cách train. Không bảo đảm thêm dữ liệu làm mọi chỉ số tăng.')
    p('Tài liệu đối chiếu: model card MiniLM https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2; '
      'model card DeBERTa https://huggingface.co/microsoft/deberta-v3-base; hướng dẫn fine tuning https://huggingface.co/docs/transformers/training; '
      'tách dữ liệu và tránh leakage https://scikit-learn.org/stable/common_pitfalls.html. '
      'Số liệu và biểu đồ được tạo lại bằng scripts/summarize_model_comparison.py từ các lượt đã lưu.')
