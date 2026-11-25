"""Package the verified numbered notebooks, report and report-to-experiment guide."""
import hashlib,json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'kaggle/teacher_experiments'


def main():
    validation=json.loads((DEST/'validation.json').read_text())
    assert validation['passed'] and validation['notebook_count']==11
    index=json.loads((DEST/'experiment_index.json').read_text())
    by_number={r['file'][:2]:r['file'] for r in index}
    rows=json.loads((ROOT/'reports/all_trials_comparison.json').read_text())
    mapping=[]
    models={'tfidf_logreg':'01','tfidf_dummy':'01','tfidf_svm':'02','tfidf_nb':'03','minilm_logreg':'04',
            'finetuned_minilm':'05','minilm_tfidf_blend':'06','joint_deberta':'07','joint_minilm':'08',
            'real_only':'08','custom_ce':'09','custom_pairwise':'09'}
    for n,row in enumerate(rows,1):
        number=models[row['model']]
        mapping.append({'record':n,'environment':row['environment'],'trial':row['trial_label'],
            'train_source':row.get('train_source','joint'),'test_source':row['test_source'],
            'policy':row.get('policy',''),'seed':row.get('seed'),'f1':row['f1'],
            'rerun_method_notebook':by_number[number],'replay_record_notebook':by_number['10']})
    (DEST/'report_experiment_map.json').write_text(json.dumps(mapping,indent=2,ensure_ascii=False)+'\n')
    lines=['# Match every report result to a runnable notebook','',
        'Each row below identifies a recorded result. Upload the linked method notebook to train the method again; notebook 10 displays the recorded evidence without training.',
        'A fresh run can differ from a historical score because hardware and software differ. Earlier Kaggle and Mac measurements remain separate.',
        'Scouts, weighting, contraction augmentation and five-fold CV are all in 05 (and as prerequisites in 06). Both seeds and all completed custom ablations are in 09. Interrupted FP32/timing-only attempts are documented in 10, with no invented classifier score.',
        '', '| Record | Original environment | Trial | Train → test | Policy / seed | F1 | Train again |',
        '|---|---|---|---|---|---|---|']
    for r in mapping:
        lines.append(f"| {r['record']} | {r['environment']} | {r['trial']} | {r['train_source']} → {r['test_source']} | {r['policy']} / {r['seed'] or ''} | {r['f1']:.4f} | [{r['rerun_method_notebook'][:2]}]({r['rerun_method_notebook']}) |")
    (DEST/'REPORT_EXPERIMENT_MAP.md').write_text('\n'.join(lines)+'\n')
    lines=['# Validation of the teacher collection','',
        'All 11 notebooks were executed locally, cell by cell, in separate temporary workspaces. Completed model weights and saved predictions were reused; training was not repeated.',
        'The checks cover embedded-source bootstrap, fresh data cleaning from locally copied pinned raw files, partition separation, recomputing statistics from predictions/confusion counts, figure generation, output packaging and inference demonstrations.',
        'Fresh Kaggle package installation, network downloads and GPU training were not executed for this new collection. The training modules reuse the completed project implementations; numerical identity across hardware is not guaranteed.',
        'Notebook outputs show LOCAL QA explicitly. On Kaggle, Run All defaults to fresh training in 01–09. Notebooks 00 and 10 do not train.',
        '', '| Notebook | Code cells executed | Results checked | With per-post checks | Figures |', '|---|---|---|---|---|']
    for r in validation['notebooks']:
        assert r['passed']
        lines.append(f"| {r['file']} | {r['executed_code_cells']} | {r['result_rows']} | {r['prediction_rows_checked']} | {r['figure_count']} |")
    (DEST/'VALIDATION.md').write_text('\n'.join(lines)+'\n')
    readme=DEST/'README.md';text=readme.read_text()
    if 'REPORT_EXPERIMENT_MAP.md' not in text:
        text+='\nĐối chiếu từng dòng trong 83 kết quả: [REPORT_EXPERIMENT_MAP.md](REPORT_EXPERIMENT_MAP.md). ZIP còn kèm báo cáo Word chính thức 18 trang.\n'
        readme.write_text(text)
    for r in index:r['sha256']=hashlib.sha256((DEST/r['file']).read_bytes()).hexdigest()
    (DEST/'experiment_index.json').write_text(json.dumps(index,indent=2,ensure_ascii=False)+'\n')
    files=[p for p in sorted(DEST.iterdir()) if p.is_file() and p.name!='SHA256SUMS.txt']
    report=ROOT/'reports/bao_cao_do_an_hoc_may.docx'
    sums=[hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name for p in files+[report]]
    (DEST/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n')
    archive=ROOT/'kaggle/teacher_experiments.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for p in files+[DEST/'SHA256SUMS.txt',report]:z.write(p,'teacher_experiments/'+p.name)
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert sum(n.endswith('.ipynb') for n in z.namelist())==11
    print('Packaged',archive,'MB',round(archive.stat().st_size/1e6,2),'mapped records',len(mapping))


if __name__=='__main__':main()
