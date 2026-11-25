"""A short notebook that only blends existing outputs; it never trains an encoder."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def cell(kind,text):
    data={'cell_type':kind,'metadata':{},'id':hashlib.sha256(text.encode()).hexdigest()[:12],'source':text.splitlines(keepends=True)}
    if kind=='code':data.update(execution_count=None,outputs=[])
    cells.append(data)
def md(s):cell('markdown',s)
def code(s):cell('code',s)
md('''# Tối ưu F1 bằng các kết quả đã huấn luyện

Notebook này không huấn luyện lại MiniLM. Nó chọn trọng số trộn MiniLM và TF-IDF Logistic Regression cùng ngưỡng bằng OOF train,
sau đó đánh giá cấu hình đã cố định trên test. Một mức tăng trên Mac không bảo đảm tăng trên phiên Kaggle khác.

Nếu tiếp tục đúng session cũ, chỉ Run All. Nếu mở session mới, chọn **Add Input** và thêm **output của notebook Kaggle đã chạy**
(thư mục Project có reports/experiments, reports/fine_tuning và models). Bản ZIP cũ 171,7 MB không có OOF của LR,
nên riêng ZIP đó chưa đủ. Notebook tự tìm bộ kết quả cũ và chỉ copy các tệp cần thiết sang /kaggle/working.
GPU không bắt buộc; phần inference kiểm tra model dùng CPU. Giữ riêng các số đo cũ và mới.
''')
code('''import os,sys,json,shutil,subprocess,importlib.util
from pathlib import Path
from IPython.display import display,Image
IS_KAGGLE=Path('/kaggle/working').exists()
PROJECT=Path(os.environ.get('FORUM_PROJECT_DIR','/kaggle/working/Project' if IS_KAGGLE else str(Path.cwd()/'Project'))).resolve()
needed=Path('reports/experiments/oof_stanford_tfidf_logreg.csv')
if not (PROJECT/needed).exists():
    previous_project=os.environ.get('FORUM_PREVIOUS_PROJECT_DIR')
    matches=([Path(previous_project)/needed] if previous_project else
             (list(Path('/kaggle/input').rglob('oof_stanford_tfidf_logreg.csv')) if IS_KAGGLE else []))
    if len(matches)!=1:
        raise RuntimeError('Attach exactly one previous notebook output containing Project/reports/experiments/oof_stanford_tfidf_logreg.csv. No encoder retraining is performed.')
    previous=matches[0].parents[2]
    for source in ['upenn','stanford']:
        for relative in [f'reports/experiments/oof_{source}_tfidf_logreg.csv',
                         *[f'reports/experiments/predictions_{source}_{target}_tfidf_logreg.csv' for target in ['upenn','stanford']],
                         f'models/{source}__tfidf_logreg.joblib']:
            target=PROJECT/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(previous/relative,target)
        for relative in [f'reports/fine_tuning/{source}',f'models/urgency_minilm_{source}']:
            shutil.copytree(previous/relative,PROJECT/relative,dirs_exist_ok=True)
    (PROJECT/'reports/fine_tuning/results.json').write_text(json.dumps([
        r for source in ['upenn','stanford'] for r in json.loads((PROJECT/f'reports/fine_tuning/{source}/test_results.json').read_text())]))
for folder in ['scripts','reports/figures','.cache/tmp']:(PROJECT/folder).mkdir(parents=True,exist_ok=True)
os.environ.update(PROJECT_ROOT=str(PROJECT),HF_HOME=str(PROJECT/'.cache/huggingface'),
    MPLCONFIGDIR=str(PROJECT/'.cache/matplotlib'),TMPDIR=str(PROJECT/'.cache/tmp'),
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
required={'torch':'torch>=2.8','transformers':'transformers>=4.55,<6','sklearn':'scikit-learn>=1.6,<2','matplotlib':'matplotlib','joblib':'joblib'}
missing=[package for module,package in required.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
print('Existing experiment folder:',PROJECT)
print('No encoder training. Only OOF selection, test evaluation, packaging and offline inference.')
''')
for name in ['prepare_data.py','train.py','blend_trained_models.py','standalone_blend.py','package_blended_models.py']:
    md('## '+name+'\n')
    source=(ROOT/'scripts'/name).read_text();literal="r'''"+source+"'''"
    code('(PROJECT/"scripts"/'+repr(name)+').write_text('+literal+',encoding="utf-8")\n')
md('''## Chọn cấu hình và kiểm tra kết quả

21 trọng số từ 0 đến 1; F1 OOF quyết định cấu hình trước khi đọc test. CV của base models có tham gia chọn mô hình/checkpoint,
nên kết quả OOF này là development diagnostics, không phải nested CV độc lập. Các test đã được xem trong phát triển;
so sánh mới là exploratory. Không tìm ngưỡng tối ưu trên test.
''')
code('''for script in ['blend_trained_models.py','package_blended_models.py']:
    subprocess.run([sys.executable,str(PROJECT/'scripts'/script)],cwd=PROJECT,check=True)
fine=json.loads((PROJECT/'reports/fine_tuning/results.json').read_text())
results=json.loads((PROJECT/'reports/ensembles/results.json').read_text())
for row in results:
    before=next(r for r in fine if r['train_source']==row['train_source'] and r['test_source']==row['test_source'])
    print(f"{row['train_source']} → {row['test_source']}: MiniLM={before['f1']:.4f}, blend={row['f1']:.4f}, delta={row['f1']-before['f1']:+.4f}, weight={row['minilm_weight']:.2f}, threshold={row['threshold']:.4f}")
    print('Precision:',row['precision'],'Recall:',row['recall'],'Confusion:',row['confusion_matrix'])
display(Image(filename=str(PROJECT/'reports/figures/blend_oof_search.png')))
''')
md('## Nhập câu mới và tải model\n\nSửa TEXT_TO_PREDICT để thử. Mỗi thư mục hybrid chứa đủ weights, tokenizer, LR và predict.py.\n')
code('''TEXT_TO_PREDICT='I cannot submit my assignment and the deadline is tonight. Please help!'
subprocess.run([sys.executable,str(PROJECT/'models/urgency_hybrid_stanford/predict.py'),'--text',TEXT_TO_PREDICT],check=True)
from zipfile import ZipFile,ZIP_DEFLATED
archive=PROJECT/'urgency_hybrid_and_statistics.zip'
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
    for folder in [PROJECT/'models/urgency_hybrid_upenn',PROJECT/'models/urgency_hybrid_stanford',PROJECT/'reports/ensembles']:
        for file in folder.rglob('*'):
            if file.is_file():z.write(file,str(file.relative_to(PROJECT)))
    z.write(PROJECT/'reports/figures/blend_oof_search.png','reports/figures/blend_oof_search.png')
print('Download:',archive)
''')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
target=ROOT/'kaggle/optimize_existing_models.ipynb';target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
print('Optimization notebook created:',target)
