"""Build a self-contained notebook: embedded scripts, pinned inputs, training and displayed evidence."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    files=['prepare_data.py','train.py','finetune_minilm.py','plot_results.py','plot_finetuning.py','predict_urgency.py',
           'standalone_inference.py','package_finetuned_models.py','blend_trained_models.py','standalone_blend.py','package_blended_models.py']
    payload={name:(ROOT/'scripts'/name).read_text(encoding='utf-8') for name in files}
    cells=[]
    def cell(kind,text):
        item={'cell_type':kind,'id':hashlib.sha256((kind+text).encode()).hexdigest()[:12],'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code':item.update(execution_count=None,outputs=[])
        cells.append(item)
    md=lambda s:cell('markdown',s)
    code=lambda s:cell('code',s)
    md('''# Ưu tiên phản hồi bài đăng sinh viên bằng học máy

Notebook tái tạo dữ liệu, baseline TF-IDF, MiniLM frozen và MiniLM fine-tuned. Có đủ Precision, Recall, F1, Macro-F1, AP, ROC-AUC, P@20/R@20, ma trận nhầm lẫn và phân tích lỗi.

**Kaggle:** bật Internet và chọn GPU trong Settings trước Run All. Không cần Google Drive, tài khoản Hugging Face hoặc API key. MiniLM được tải ở revision cố định. Chạy đủ hai nguồn và 5-fold sẽ lâu hơn bản chỉ trích embeddings.

GPU T4 ×2 dùng được. Trainer hiện dùng một GPU cho mỗi lượt chạy, chưa phân tán trên cả hai GPU. Trên T4, fine-tuning dùng FP32; BF16 chỉ bật khi CUDA hỗ trợ phần cứng trực tiếp.

Test đã được xem trong các thí nghiệm trước, vì vậy các so sánh mở rộng được ghi nhận là **exploratory**. Không chọn epoch, cấu hình hoặc ngưỡng bằng test. Nhãn là `int(urgency > 4)`; phạm vi tiếng Anh.
''')
    code('''import os, sys, json, subprocess, importlib.util
from pathlib import Path
from IPython.display import display, Markdown, HTML, Image

IS_KAGGLE = Path('/kaggle/working').exists()
PROJECT = Path(os.environ.get('FORUM_PROJECT_DIR', '/kaggle/working/Project' if IS_KAGGLE else str(Path.cwd() / 'Project'))).resolve()
PROJECT.mkdir(parents=True, exist_ok=True)
for folder in ['scripts','data/raw','data/processed','models','reports/experiments','reports/fine_tuning','reports/figures','.cache/tmp']:
    (PROJECT / folder).mkdir(parents=True, exist_ok=True)
os.environ.update(PROJECT_ROOT=str(PROJECT), HF_HOME=str(PROJECT/'.cache/huggingface'),
    HF_HUB_CACHE=str(PROJECT/'.cache/huggingface/hub'), HF_XET_CACHE=str(PROJECT/'.cache/huggingface/xet'),
    TORCH_HOME=str(PROJECT/'.cache/torch'), MPLCONFIGDIR=str(PROJECT/'.cache/matplotlib'),
    TMPDIR=str(PROJECT/'.cache/tmp'), PIP_CACHE_DIR=str(PROJECT/'.cache/pip'),
    TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_TELEMETRY='1', PYTORCH_ENABLE_MPS_FALLBACK='1',
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

# Default: train from the pinned data. Local QA may explicitly reuse completed artifacts.
RUN_TRAINING = os.environ.get('FORUM_REUSE_EXISTING', '0') != '1'
MAX_EPOCHS = 3
PATIENCE = 1
FOLDS = 5
BATCH_SIZE = 64
MAX_LENGTH = 256
print('Project:', PROJECT)
print('Execution mode:', 'TRAIN MODELS' if RUN_TRAINING else 'REUSE COMPLETED TRAINING ARTIFACTS (explicit local QA mode)')
''')
    md('''## Thư viện và thiết bị

Không tải hoặc cài lại PyTorch nếu môi trường đã có. Khi thiếu thư viện, cell dưới cài vào môi trường notebook. Trên Mac, dùng môi trường `.venv` của Project; trên Kaggle, dùng môi trường notebook của Kaggle. Các phiên bản thực tế được in và ghi lại.
''')
    code('''required = {'transformers':'transformers==4.57.6', 'sentence_transformers':'sentence-transformers>=5,<6',
            'sklearn':'scikit-learn>=1.7,<2', 'matplotlib':'matplotlib>=3.10,<4', 'torch':'torch>=2.8'}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
import torch, transformers, sklearn, numpy as np
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
USE_BF16 = (DEVICE == 'mps') or (DEVICE == 'cuda' and torch.cuda.is_bf16_supported(including_emulation=False))
if IS_KAGGLE and RUN_TRAINING and DEVICE != 'cuda':
    raise RuntimeError('Enable GPU in Kaggle Settings, then Run All again.')
versions = {'python':sys.version, 'torch':torch.__version__, 'transformers':transformers.__version__,
            'sklearn':sklearn.__version__, 'numpy':np.__version__, 'device':DEVICE, 'bf16':USE_BF16,
            'cuda_devices_visible':torch.cuda.device_count() if DEVICE == 'cuda' else 0,
            'cuda_device_used':torch.cuda.get_device_name(0) if DEVICE == 'cuda' else None,
            'training_gpu_count':1 if DEVICE == 'cuda' else 0}
print(json.dumps(versions, indent=2))
(PROJECT/'reports/notebook_environment.json').write_text(json.dumps(versions, indent=2))
''')
    md('''## Mã nguồn được đóng gói trong notebook

Các cell dưới ghi từng script đầy đủ, không phụ thuộc repository cá nhân. Mã được hiển thị theo dòng để giảng viên đọc thuật toán và quy trình. SHA-256 được in để kiểm tra phiên bản mã.
''')
    code('SOURCES = {}\n')
    for name,source in payload.items():
        md('### Script '+name+'\n')
        literal="r'''"+source+"'''" if "'''" not in source else repr(source)
        code('SOURCES['+repr(name)+'] = '+literal+'\n'+
             '(PROJECT/"scripts"/'+repr(name)+').write_text(SOURCES['+repr(name)+'], encoding="utf-8")\n')
    code('''import hashlib
print(json.dumps({name:hashlib.sha256(source.encode()).hexdigest() for name,source in SOURCES.items()},indent=2))
def run_script(name, *args):
    subprocess.run([sys.executable, str(PROJECT/'scripts'/name), *map(str,args)], cwd=PROJECT, check=True, env=os.environ.copy())
def read_json(relative):
    return json.loads((PROJECT/relative).read_text(encoding='utf-8'))
def show_table(headers, rows):
    import html
    markup='<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'
    markup+=''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)
    display(HTML(markup+'</tbody></table>'))
''')
    md('''## Hai nguồn dữ liệu và làm sạch

UPenn và Stanford MOOCPosts tải từ kho bổ trợ nghiên cứu ở commit cố định. Kiểm tra hash gốc, nhãn 1–7, dòng thiếu trường và trùng chính xác. Giữ nguyên raw CSV. Lý do loại ghi trong `rejected_rows.csv`.
''')
    code('''run_script('prepare_data.py', '--download')
audit=read_json('reports/data_audit.json')
show_table(['Source','Raw rows','Clean rows','Urgent','Not urgent'],
    [[s,d['counts']['raw_rows'],d['counts']['final_rows'],d['label_counts']['1'],d['label_counts']['0']]
     for s,d in audit['datasets'].items()])
print(json.dumps(audit['checks'], indent=2))
''')
    md('''## Baseline và tập chia cố định

TF-IDF + LR/SVM/Multinomial NB, Dummy và frozen MiniLM + LR. UPenn chia theo student ID; Stanford chia phân tầng vì bản CSV không có ID tác giả/luồng. Tuning baseline dùng 3-fold trong train; test giữ nguyên cho so sánh fine-tuning.
''')
    code('''if RUN_TRAINING:
    run_script('train.py')
run_script('plot_results.py')
splits=read_json('reports/experiments/split_info.json')
show_table(['Source','Train','Test','Urgent train','Urgent test'],
    [[s,d['train'],d['test'],d['train_positives'],d['test_positives']] for s,d in splits.items()])
baseline=read_json('reports/experiments/results.json')
show_table(['Train','Test','Method','Precision','Recall','F1','Macro F1','AP'],
    [[r['train_source'],r['test_source'],r['model'],*[round(r[k],4) for k in ['precision','recall','f1','macro_f1','average_precision']]] for r in baseline])
''')
    md('''## Fine tuning MiniLM và kiểm tra các kỹ thuật

Toàn bộ encoder và head phân loại được cập nhật. Ba cấu hình scout: cross-entropy thường; class-weighted loss; class-weighted loss cộng một biến thể mở rộng contractions khi văn bản thay đổi. Biến thể chỉ tạo từ train, giữ nguyên phủ định và deadline. Đây là augmentation bảo thủ, không phải sinh paraphrase bằng LLM.

Scout chọn cấu hình bằng F1 validation sau chọn ngưỡng; checkpoint theo AP validation. Sau đó chạy 5-fold trên train nguồn, khởi tạo lại cùng pretrained checkpoint mỗi fold. Early stopping dùng validation của fold nên các điểm CV này là **development diagnostics**, không phải đánh giá nested CV không thiên lệch. Test không dùng chọn mô hình.

Model cuối train trên toàn bộ train nguồn với số epoch bằng trung vị best epoch của 5 fold; ngưỡng lấy từ điểm OOF. Không early-stop bằng test.
''')
    code('''if RUN_TRAINING:
    arguments=['--stage','all','--epochs',MAX_EPOCHS,'--patience',PATIENCE,'--folds',FOLDS,
               '--batch-size',BATCH_SIZE,'--max-length',MAX_LENGTH]
    if USE_BF16:
        arguments.append('--bf16')
    run_script('finetune_minilm.py', *arguments)
run_script('plot_finetuning.py')
run_script('package_finetuned_models.py')
for source in ['upenn','stanford']:
    selection=read_json(f'reports/fine_tuning/{source}/selection.json')
    print(source, 'selected:', selection['selected'])
    show_table(['Configuration','Val AP','Val tuned F1','Best epoch','Epochs run','Augmented posts'],
        [[r['name'],round(r['validation_ap'],4),round(r['validation_f1_tuned'],4),r['best_epoch'],r['epochs_run'],r['augmented_posts']]
         for r in selection['scout_runs']])
    cv=read_json(f'reports/fine_tuning/{source}/cv.json')
    show_table(['Fold','Best epoch','Val AP','Val tuned F1','Training seconds'],
        [[i+1,r['best_epoch'],round(r['validation_ap'],4),round(r['validation_f1_tuned'],4),round(r['seconds'],1)]
         for i,r in enumerate(cv['folds'])])
    print('OOF:',json.dumps(cv['oof_metrics'],indent=2))
''')
    md('''## Tối ưu bằng các mô hình đã huấn luyện

Trộn điểm MiniLM fine-tuned và TF-IDF Logistic Regression, không train lại encoder. Chọn trọng số từ 0 đến 1, bước 0,05,
và ngưỡng bằng dự đoán OOF của train nguồn. Cấu hình được lưu trước khi đọc test. Đây vẫn là development diagnostics,
không phải nested CV. Không cam kết đạt cùng mức tăng của phiên Mac trên phiên Kaggle.
''')
    code('''run_script('blend_trained_models.py')
run_script('package_blended_models.py')
blended=read_json('reports/ensembles/results.json')
for source in ['upenn','stanford']:
    print(source,read_json(f'reports/ensembles/{source}/decision_config.json'))
display(Image(filename=str(PROJECT/'reports/figures/blend_oof_search.png')))
''')
    md('''## Tất cả số đo kiểm tra và so sánh

Precision/Recall/F1 tính cho lớp ưu tiên. Macro-F1 trung bình hai lớp; AP và ROC-AUC dùng điểm liên tục. P@20 và R@20 đánh giá hàng đợi 20 bài đầu. Không chọn cấu hình bằng các bảng test này.
''')
    code('''fine=read_json('reports/fine_tuning/results.json')
all_results=baseline+fine+blended
keys=['precision','recall','f1','macro_f1','accuracy','average_precision','roc_auc','precision_at_20','recall_at_20']
for source,target in [('upenn','upenn'),('stanford','stanford'),('upenn','stanford'),('stanford','upenn')]:
    display(Markdown(f'### {source} → {target}'))
    selected=[r for r in all_results if r['train_source']==source and r['test_source']==target]
    show_table(['Method',*keys],[[r['model'],*[round(r[k],4) for k in keys]] for r in selected])
    for r in selected:
        print(r['model'],'confusion matrix [[TN, FP], [FN, TP]]:',r['confusion_matrix'])
''')
    md('''## Biểu đồ từ dữ liệu thực nghiệm

Các hình được tạo từ JSON/CSV đã lưu: learning curves, 5-fold, so sánh P/R/F1, ma trận nhầm lẫn, Precision–Recall, chuyển nguồn và chọn ngưỡng. Không có số liệu mẫu thay cho kết quả training.
''')
    code('''for name in ['ft_learning_curves.png','ft_cv.png','ft_comparison.png','ft_confusion.png','ft_pr.png',
             'upenn_metrics.png','stanford_metrics.png','transfer.png','confusion_four.png','upenn_pr.png','stanford_pr.png','threshold.png']:
    display(Markdown('**'+name+'**'))
    display(Image(filename=str(PROJECT/'reports/figures'/name)))
''')
    md('''## Dự đoán sai và giới hạn

Ví dụ lấy từ test fine-tuned, không sửa nhãn để tăng điểm. Chênh lệch từ một seed chưa chứng minh ý nghĩa thống kê. MiniLM có thể cắt bài dài; nhãn urgency không đồng nhất với có câu hỏi. Chưa đánh giá tiếng Việt hoặc hiệu quả giảng viên thực tế.
''')
    code('''import csv
for source in ['upenn','stanford']:
    with (PROJECT/f'reports/fine_tuning/{source}/predictions_{source}.csv').open() as handle:
        predictions=list(csv.DictReader(handle))
    wrong=[r for r in predictions if r['label']!=r['predicted']]
    display(Markdown(f'### {source}: {len(wrong)} wrong predictions'))
    show_table(['Post ID','True','Predicted','Score','Text excerpt'],
        [[r['post_id'],r['label'],r['predicted'],round(float(r['score']),4),r['text'][:240]] for r in wrong[:8]])
''')
    md('''## Dùng thư mục model để dự đoán câu mới

Đổi `TEXT_TO_PREDICT` rồi chạy lại cell. Model và tokenizer được đọc từ thư mục local; không cần gọi AI API. Score được so với ngưỡng lưu trong decision_config.json.
''')
    code('''TEXT_TO_PREDICT = 'I cannot submit my assignment and the deadline is tonight. Please help!'
MODEL_SOURCE = 'stanford'
run_script('predict_urgency.py', '--model',PROJECT/f'models/urgency_minilm_{MODEL_SOURCE}', '--text',TEXT_TO_PREDICT)
print(read_json(f'reports/fine_tuning/{MODEL_SOURCE}/export_check.json'))
run_script('standalone_blend.py','--model',PROJECT/f'models/urgency_hybrid_{MODEL_SOURCE}','--text',TEXT_TO_PREDICT)
''')
    md('''## Tải model và bằng chứng

File ZIP chứa model fine-tuned, model hybrid độc lập, cấu hình, tokenizer, metrics, fold IDs và hình. Các CSV raw vẫn có URL và hash trong audit; không đưa raw vào ZIP model.
''')
    code('''from zipfile import ZipFile, ZIP_DEFLATED
archive=PROJECT/'urgency_minilm_and_statistics.zip'
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
    for folder in [PROJECT/'models/urgency_minilm_upenn',PROJECT/'models/urgency_minilm_stanford',
                   PROJECT/'models/urgency_hybrid_upenn',PROJECT/'models/urgency_hybrid_stanford',
                   PROJECT/'reports/fine_tuning',PROJECT/'reports/ensembles',PROJECT/'reports/figures',PROJECT/'reports/experiments']:
        for file in folder.rglob('*'):
            if file.is_file():
                z.write(file,str(file.relative_to(PROJECT)))
    for source in ['upenn','stanford']:
        file=PROJECT/f'models/{source}__tfidf_logreg.joblib'
        z.write(file,str(file.relative_to(PROJECT)))
print('Download from Kaggle Output:',archive,'MB:',round(archive.stat().st_size/1024**2,1))
''')
    md('''## Nguồn

- Švábenský và cộng sự, *Towards Generalizable Detection of Urgency of Discussion Forum Posts*: https://arxiv.org/abs/2307.07614
- Dữ liệu và mã bổ trợ: https://github.com/pcla-code/forum-posts-urgency
- Agrawal và Paepcke, *The Stanford MOOC Posts Dataset*, dẫn trong README kho trên.
- MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Transformers text classification: https://huggingface.co/docs/transformers/tasks/sequence_classification
''')
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'},'kaggle':{'accelerator':'gpu','isInternetEnabled':True}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'kaggle/urgency_minilm_training.ipynb';target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Notebook created:',target,'cells:',len(cells),'bytes:',target.stat().st_size)

if __name__=='__main__':main()
