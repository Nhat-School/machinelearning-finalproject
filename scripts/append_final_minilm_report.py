"""Append measured final MiniLM ablations to the same Vietnamese report."""
import json
from pathlib import Path
from build_report import table,figure as standard_figure
from docx.shared import Cm,Pt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'reports/final_minilm_trial'
def load(path):return json.loads(path.read_text())
def figure(d,name,caption,width=17):
    if width==17:return standard_figure(d,name,caption)
    paragraph=d.add_paragraph();paragraph.paragraph_format.keep_with_next=True
    paragraph.add_run().add_picture(str(ROOT/'reports/figures'/name),width=Cm(width))
    paragraph=d.add_paragraph(caption)
    for run in paragraph.runs:run.italic=True;run.font.size=Pt(8.5)
def append_final_minilm(d):
    if not (BASE/'summary.json').exists():return
    data=load(BASE/'summary.json');results=data['results'];contexts=data['contexts'];protocols=data['protocols']
    full=(BASE/'results.json').exists();selection=load(BASE/'selection.json') if full else None
    chosen=selection['selected_custom_configuration'] if full else None
    names={'real_only':'B: hai nguồn','custom_ce':'C: custom CE','custom_pairwise':'C: custom pair'}
    def p(text):d.add_paragraph(text)
    def page(title):d.add_page_break();d.add_heading(title,1)
    def f(v):return f'{v:.4f}'
    def title(r):return names[r['arm']]+' / '+str(r['seed'])
    page('Thử nghiệm cuối: MiniLM học chung và dữ liệu ngữ cảnh')
    p('Mục tiêu là giảm phân loại sai bài mới, đặc biệt khi vấn đề đã được giải quyết hoặc cập nhật nhưng vẫn bị chặn. '
      'Các nhánh giữ cùng pretrained MiniLM L6 v2, tokenizer, split thật, optimizer và tiêu chí chọn checkpoint. '
      'Bắt đầu từ pretrained gốc, không tiếp tục checkpoint nguồn riêng đã từng nhìn thấy các hàng nay dành cho calibration và validation.')
    table(d,['Nguồn','Fit','Calibration','Validation','Test / challenge'],[
        ['UPenn','2.257 (189 ưu tiên)','275 (25)','270 (23)','691 (61)'],
        ['Stanford','18.604 (3.535)','2.326 (442)','2.326 (442)','5.814 (1.105)'],
        ['Custom binary draft','82 (48 ưu tiên)','Không dùng','30 (18)','24 (14)']], [3,3.5,3.5,3.5,3.5])
    p('CustomForum gồm 160 câu tổng hợp do trợ lý soạn. Chỉ 136 hàng có nhãn nhị phân; các hàng mơ hồ hoặc ngoài phạm vi không bị ép thành NORMAL. '
      f"Hiện {data['custom_reviewed_count']} hàng được con người duyệt. Các nhánh C dùng nhãn dự thảo như weak supervision trong thử nghiệm được yêu cầu, "
      'không chuyển worksheet sang approved. Không coi accuracy trên các câu này là độ chính xác ngoài thực tế.')
    table(d,['Nhánh','Cập nhật trọng số','Vai trò'],[
        ['B real_only','100% loss từ hai nguồn thật','Đối chứng cùng kiến trúc'],
        ['C custom_ce','80% loss thật + 20% CE custom','Kiểm tra tác động nhãn tổng hợp'],
        ['C custom_pairwise','Như CE, thêm 0,25 loss thứ tự bên trong phần custom','Đẩy điểm đang bị chặn cao hơn cùng tình huống đã giải quyết']], [4,7,6])
    p('Giữ ba trạng thái của mỗi nhóm trong cùng partition. Train có 24 bộ ba hoàn chỉnh; validation có 9; challenge có 7. '
      'Mỗi bước custom lấy hai bộ ba và bổ sung đến 12 hàng. Loss custom cân bằng hai lớp; loss thật cân bằng tổng đóng góp hai nguồn, giữ tỷ lệ lớp tự nhiên. '
      'Hệ số pairwise thực tế trong tổng loss là 0,20 × 0,25 = 0,05; không dùng quy tắc từ khóa để ép nhãn khi dự đoán.')
    p('Tối đa 4 epoch, patience 2, learning rate 2e-5, batch 64, chiều dài 256 token. Checkpoint tối đa AP validation trung bình hai nguồn thật. '
      'Platt calibration chỉ fit trên calibration thật riêng, cân bằng nguồn. Ngưỡng binary 0,5 được cố định trước test; khoảng 0,2 đến 0,8 chuyển Review. '
      'Chỉ số binary vẫn tính trên mọi bài, gồm cả bài Review.')
    page('Precision, Recall, F1 và lỗi trên hai nguồn thật')
    table(d,['Nhánh / seed','Test','P','R','F1','Acc','FP / FN'],[
        [title(r),r['test_source'],*[f(r[k]) for k in ['precision','recall','f1','accuracy']],
         f"{r['confusion_matrix'][0][1]} / {r['confusion_matrix'][1][0]}"]
        for r in results if r['test_source'] in ['upenn','stanford']], [4.3,2,2,2,2,2,2.7])
    figure(d,'final_minilm_metrics.png','Hình 33. Số đo từ toàn bộ dự đoán test đã lưu. Các pipeline cũ có calibration và thành phần train khác nhau.')
    p('Precision/Recall/F1 tính cho lớp ưu tiên; Accuracy tính tất cả bài. UPenn chỉ có 61 bài ưu tiên trên 691 test, nên Accuracy cao có thể che việc bỏ sót lớp hiếm. '
      'Không chọn seed, checkpoint hoặc ngưỡng bằng điểm test. Các test lịch sử đã được xem trước đó; kết quả là so sánh phát triển, chưa phải test ngoài mới.')
    page('Ma trận nhầm lẫn và diễn biến học')
    figure(d,'final_minilm_confusion.png','Hình 34. Seed 42: hàng là nhãn gốc, cột là dự đoán; góc phải trên là báo nhầm, trái dưới là bỏ sót.',width=15)
    p('Mỗi ma trận đối soát lại từ CSV điểm và nhãn trên cùng post ID test. Số bài không bị giảm vì chính sách Review. '
      'Giữ đủ kết quả của cả hai cấu hình custom, kể cả cấu hình không được chọn cho lần xác nhận.')
    page('Lựa chọn cấu hình và kiểm tra hai seed')
    if full:
        frozen=load(BASE/'frozen_recommendation.json')
        p('Chọn custom bằng validation: AP trung bình hai nguồn thật không giảm quá 0,03 so với B cùng seed, '
          'sau đó tối đa balanced accuracy trên custom validation; nếu bằng nhau chọn AP thật cao hơn. '
          'Chỉ đề xuất custom khi điều kiện giữ AP và cải thiện context lặp lại ở seed 43. Quy tắc này ghi trong experiment_plan.json trước test mới; '
          'test B đã được xem trước khi huấn luyện C, nhưng quy tắc và cấu hình C không đổi theo điểm B test.')
        cards=[selection['reference'],*selection['candidates']]
        table(d,['Seed 42','AP thật trung bình','BA custom val','Đúng đủ ba trạng thái'],[
            [names[c['arm']],f(c['mean_real_validation_ap']),f(c['context_validation']['balanced_accuracy']),
             str(c['context_validation']['all_three_states_correct'])+' / '+str(c['context_validation']['complete_scenario_groups'])] for c in cards],[4.4,4,4,4.6])
        p(f"Cấu hình custom đã chọn: {chosen}. Nhánh được quy tắc validation đề xuất: {frozen['preferred_trial_arm']}; "
          'seed triển khai minh họa luôn là 42, không lấy seed test cao hơn. Đây là đề xuất trong phạm vi thử nghiệm, không bảo đảm thắng mọi tiêu chí.')
    seeds=load(BASE/('two_seed_summary.json' if full else 'control_two_seed_summary.json'))
    table(d,['Nhánh','Nguồn','F1 seed 42','F1 seed 43','Trung bình ± SD'],[
        [names[arm],source,f(values['f1']['values_by_seed']['42']),f(values['f1']['values_by_seed']['43']),
         f(values['f1']['mean'])+' ± '+f(values['f1']['std_across_two_seeds'])]
        for arm,domains in seeds.items() for source,values in domains.items() if source in ['upenn','stanford']],[4,2.5,3,3,4.5])
    p('Độ lệch chuẩn trên hai khởi tạo chỉ mô tả dao động quan sát được; không phải khoảng tin cậy hoặc kiểm định thống kê. '
      'Các nhánh mới dùng hold-out phát triển và hai seed; CV năm fold ở phần MiniLM cũ là thí nghiệm khác, không được gán sang nhánh mới.')
    figure(d,'final_minilm_learning.png','Hình 35. Loss train và AP validation thật từ log từng epoch. Loss giảm không tự bảo đảm cải thiện ngữ cảnh hoặc test.')
    page('Tình huống ngữ cảnh và các lỗi còn lại')
    table(d,['Nhánh / seed','Synthetic challenge P','R','F1','Đúng đủ bộ ba'],[
        [title(r),f(r['precision']),f(r['recall']),f(r['f1']),
         str(r['all_three_states_correct'])+' / '+str(r['complete_scenario_groups'])]
        for r in results if r['test_source']=='custom_challenge'],[4.5,4,2.5,2.5,3.5])
    p('Challenge binary chỉ có 24 câu và 7 bộ ba hoàn chỉnh, chia theo nhóm tình huống. Câu train và challenge cùng do một nguồn soạn, '
      'có thể chia sẻ phong cách hoặc mẫu câu; nhóm tách rời chưa loại trừ mọi trùng gần nghĩa. '
      'Chỉ số tăng cho thấy học được hành vi trong bộ tình huống này, chưa chứng minh phân loại tốt mọi câu người dùng tự nhập.')
    selected_run=(chosen if full else 'real_only')+'_seed42'
    for state in ['blocked','resolved','still_blocked_update']:
        row=next(r for r in contexts[selected_run]['submission_examples'] if r['case_type']==state)
        p(f"Ví dụ {state}: điểm {float(row['score']):.4f}; nhãn binary {row['predicted']}; action {row['action']}. "+row['text'])
    errors=load(BASE/'error_examples.json')
    for source in ['upenn','stanford']:
        for kind in ['FP','FN']:
            rows=errors[selected_run][source][kind]
            if not rows:continue
            r=rows[0];p(f"Lỗi thật {source} {kind}, ID {r['post_id']}, nhãn {r['label']}, điểm {float(r['score']):.4f}, action {r['action']}: "+r['text'][:170])
    p('Lỗi lấy theo quy tắc cố định: FP điểm cao nhất và FN điểm thấp nhất, không phải mẫu ngẫu nhiên đại diện mọi lỗi. '
      'Toàn văn và nhãn nằm trong predictions của từng run; không sửa nhãn gốc để làm điểm đẹp hơn.')
    probes=BASE/'behavior_probe_results.json'
    if probes.exists():
        page('Câu diễn đạt mới: kiểm tra hành vi qua ứng dụng')
        checks=load(probes)
        table(d,['Mô hình','Đúng binary draft','Lỗi theo kỳ vọng draft','Câu mơ hồ / ngoài phạm vi tự xếp'],[
            [r['model_key'].replace('joint__','').replace('stanford__',''),str(r['correct_binary_probes'])+' / '+str(r['diagnostic_binary_count']),
             ', '.join(r['binary_errors']) or 'Không thấy trong 16 câu này',
             str(len(r['unlabelled_probes_automatically_classified']))+' / 4'] for r in checks],[5,3,6,3])
        p('Bộ probe gồm 16 câu có kỳ vọng binary do trợ lý soạn và 4 câu thiếu ngữ cảnh / ngoài phạm vi để quan sát action. '
          'Định nghĩa được lưu trước khi đánh giá C test; không dùng để train, chọn checkpoint, calibration hoặc chọn cấu hình. '
          'Không có trùng văn bản chính xác với CustomForum, nhưng chưa kiểm định độc lập hoặc loại trừ tương đồng ngữ nghĩa. '
          'Bảng là kiểm tra hành vi bằng API thật; không phải accuracy trên dữ liệu sinh viên thật hoặc nhãn giảng viên đã duyệt.')
        candidate=next((r for r in checks if r['model_key']=='joint__minilm_custom_trial'),None)
        if candidate:
            for row in [r for r in candidate['predictions'] if r['id'] in candidate['binary_errors']][:2]:
                p(f"Lỗi còn lại {row['id']}: kỳ vọng draft {row['expected_binary_priority']}, dự đoán {row['priority']}, điểm {row['score']:.4f}, action {row.get('action')}. "+row['text'])
            p(f"Có {len(candidate['unlabelled_probes_automatically_classified'])} / 4 probe mơ hồ hoặc ngoài phạm vi vẫn được tự động gán nhãn. "
              'Review dựa vào khoảng điểm không phải bộ phát hiện ngoài phạm vi; ứng dụng chỉ được đánh giá cho bài diễn đàn tiếng Anh.')
        if full:
            d.add_heading('Kết luận của lượt thử cuối',2)
            for source in ['upenn','stanford']:
                b=next(r for r in results if r['arm']=='real_only' and r['seed']==42 and r['test_source']==source)
                c=next(r for r in results if r['arm']==chosen and r['seed']==42 and r['test_source']==source)
                p(f"{source}: F1 đối chứng {b['f1']:.4f}, custom {c['f1']:.4f}; chênh lệch {(c['f1']-b['f1'])*100:+.2f} điểm phần trăm. "
                  f"Báo nhầm / bỏ sót đổi từ {b['confusion_matrix'][0][1]} / {b['confusion_matrix'][1][0]} sang {c['confusion_matrix'][0][1]} / {c['confusion_matrix'][1][0]}.")
            p('Đọc thay đổi lỗi và hành vi cập nhật cùng với hai seed, thay vì gọi một model tốt nhất cho mọi mục tiêu. '
              'Không điều chỉnh tiếp cấu hình bằng các kết quả test hoặc probe này trong lượt cuối.')
    page('Review, minh chứng ứng dụng và cách chạy lại')
    real_rows=[r for r in results if r['seed']==42 and r['test_source'] in ['upenn','stanford'] and r['arm'] in ['real_only',chosen]]
    table(d,['Nhánh / nguồn','Tự động','Review','Lỗi tự động','Ưu tiên bị bỏ sót tự động'],[
        [names[r['arm']]+' / '+r['test_source'],f(r['automatic_coverage']),r['review_count'],r['automatic_errors'],r['urgent_automatically_missed']] for r in real_rows],[5,2.5,2.5,3,4])
    p('Tự động là phần bài có điểm ≤ 0,2 hoặc ≥ 0,8. Review không xóa lỗi khỏi F1 và chưa được tính là đã xử lý đúng. '
      'Điểm được hiệu chỉnh trên hai nguồn thật, không phải độ bảo đảm đúng của một câu riêng hoặc một bộ phát hiện ngoài phạm vi.')
    shot=ROOT/'reports/screenshots/11_final_minilm.png'
    if shot.exists():
        d.add_picture(str(shot),width=Cm(15))
        p('Hình 36. Chrome chạy thật với model MiniLM thử nghiệm; nhãn, điểm và action được đối chiếu API. App vẫn cho chọn các mô hình cũ để so sánh.')
    p('Chạy sản phẩm: bash Project/scripts/launchers/start_app.sh, mở http://127.0.0.1:8765. Chọn MiniLM joint 2 nguồn hoặc MiniLM + CustomForum trong dropdown. '
      'Nhập câu tiếng Anh bất kỳ rồi bấm Phân loại bài đăng; không có quy tắc ép kết quả theo nút ví dụ.')
    page('Tái tạo toàn bộ lượt MiniLM cuối')
    p('Các lệnh dưới chạy từ machine. Toàn bộ dependencies, dữ liệu, model, cache, kết quả và báo cáo vẫn ở Project; không cài global. '
      'Lệnh train giữ các run đã hoàn tất; dùng Project mới nếu muốn đo lại từ đầu. Không trộn model cũ và mới vào một folder.')
    for command in [
        'bash Project/scripts/launchers/reproduce.sh --final-minilm',
        'bash Project/scripts/launchers/reproduce.sh',
        'python3 Project/scripts/verify_evidence.py']:
        d.add_paragraph(command,style='No Spacing')
    table(d,['Tệp / thư mục','Nội dung'],[
        ['Datasets/UPenn, Stanford, CustomForum','Nguồn thật, draft và worksheet nhãn riêng'],
        ['reports/final_minilm_trial','Plan, authorization, split hash, config, history, metrics, scores, errors và hai seed'],
        ['models/urgency_final_minilm_*','Weights, tokenizer, calibration, predict.py và yêu cầu thư viện'],
        ['kaggle/final_minilm_trial.ipynb','Notebook tự chứa mã; mặc định train và in số đo, ma trận, biểu đồ, lỗi'],
        ['final_minilm_models_and_statistics.zip','Các model, ba thư mục dữ liệu và thống kê để tải / lưu trữ']], [7,10])
    p('Trên Kaggle: bật Internet và T4 ×2, upload notebook mới rồi Run All. Không cần tự đính kèm hai CSV hoặc API key cho lượt draft yếu mặc định. '
      'Trainer dùng một T4; GPU thứ hai không được cộng vào VRAM. Phiên bản thư viện và thiết bị được in trong notebook; kết quả khác máy có thể dao động.')
    validation=BASE/'notebook_validation.json'
    if validation.exists():
        checked=load(validation)
        p(f"Đã thực thi local đủ {checked['executed_code_cells']} code cell, tái dùng các run train thật hoàn tất. "
          'Đây là kiểm tra notebook local, không phải lời khẳng định đã chạy notebook mới trên Kaggle GPU.')
    table(d,['Run','Epoch được giữ','Thời gian giây','Custom train'],[
        [name,pr['best_epoch'],f(pr['training_seconds']),pr['custom_training_rows']] for name,pr in protocols.items()],[7,3,4,3])
    p('Không có model nào bảo đảm đúng mọi câu. Kết luận cuối phải xét đồng thời Precision, Recall, F1, lỗi theo nhãn thật, hành vi cập nhật, '
      'chi phí Review và phạm vi dữ liệu. Bước xác nhận thực tế cần bài mới cùng định nghĩa nhãn được giảng viên duyệt độc lập.')
