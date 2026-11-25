"""Transcribe the user-supplied final Kaggle log; recompute count-based statistics.

This records an actual user-reported run, not local training. AP, ROC-AUC and
Brier scores are copied from the supplied log because per-post Kaggle files
have not been imported. Nothing in this script fits a model.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/kaggle_run'


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def counts_metrics(matrix):
    (tn, fp), (fn, tp) = matrix
    f1 = 2 * tp / (2 * tp + fp + fn)
    return {
        'precision': tp / (tp + fp), 'recall': tp / (tp + fn), 'f1': f1,
        'macro_f1': (f1 + 2 * tn / (2 * tn + fp + fn)) / 2,
        'accuracy': (tn + tp) / (tn + fp + fn + tp),
        'confusion_matrix': matrix, 'test_size': tn + fp + fn + tp,
        'positive_count': fn + tp,
    }


def main():
    entries = [
        (42, 'upenn', [[608, 22], [20, 41]], 425, {
            'average_precision': .697783382060938, 'roc_auc': .936247723132969,
            'precision_at_20': .95, 'recall_at_20': .3114754098360656, 'top20_found': 19,
            'brier_score': .049237521025866725, 'automatic_coverage': .8986975397973951,
            'review_count': 70, 'automatic_accuracy': .9677938808373591,
            'automatic_errors': 20, 'urgent_automatically_found': 26,
            'urgent_automatically_missed': 12, 'urgent_requiring_review': 23}),
        (42, 'stanford', [[4538, 171], [306, 799]], 427, {
            'average_precision': .8506134022041771, 'roc_auc': .9567657580698942,
            'precision_at_20': 1., 'recall_at_20': .01809954751131222, 'top20_found': 20,
            'brier_score': .060838059813975554, 'automatic_coverage': .8558651530787754,
            'review_count': 838, 'automatic_accuracy': .9628215434083601,
            'automatic_errors': 185, 'urgent_automatically_found': 505,
            'urgent_automatically_missed': 136, 'urgent_requiring_review': 464}),
        (43, 'upenn', [[601, 29], [20, 41]], 433, {
            'average_precision': .6872809551022474, 'roc_auc': .9320062451209993,
            'precision_at_20': .95, 'recall_at_20': .3114754098360656, 'top20_found': 19,
            'brier_score': .05216424915262062, 'automatic_coverage': .8798842257597684,
            'review_count': 83, 'automatic_accuracy': .9703947368421053,
            'automatic_errors': 18, 'urgent_automatically_found': 27,
            'urgent_automatically_missed': 12, 'urgent_requiring_review': 22}),
        (43, 'stanford', [[4533, 176], [319, 786]], 435, {
            'average_precision': .8475725351000832, 'roc_auc': .9566733193105722,
            'precision_at_20': .95, 'recall_at_20': .017194570135746608, 'top20_found': 19,
            'brier_score': .06068972307892164, 'automatic_coverage': .8565531475748194,
            'review_count': 834, 'automatic_accuracy': .9630522088353414,
            'automatic_errors': 184, 'urgent_automatically_found': 506,
            'urgent_automatically_missed': 130, 'urgent_requiring_review': 469}),
        (42, 'custom_challenge', [[3, 7], [0, 14]], 575, {
            'average_precision': .6759427814187876, 'roc_auc': .6571428571428571,
            'precision_at_20': .65, 'recall_at_20': .9285714285714286, 'top20_found': 13,
            'balanced_accuracy': .65, 'complete_scenario_groups': 7,
            'all_three_states_correct': 0, 'label_origin': ['assistant_draft_weak_label']}),
        (43, 'custom_challenge', [[3, 7], [0, 14]], 684, {
            'average_precision': .7617176682966156, 'roc_auc': .7214285714285714,
            'precision_at_20': .7, 'recall_at_20': 1., 'top20_found': 14,
            'balanced_accuracy': .65, 'complete_scenario_groups': 7,
            'all_three_states_correct': 0, 'label_origin': ['assistant_draft_weak_label']}),
    ]
    results = []
    for seed, source, matrix, line, extras in entries:
        record = {'run': f'real_only_seed{seed}', 'arm': 'real_only', 'seed': seed,
                  'train_source': 'joint', 'test_source': source, 'model': 'joint_minilm',
                  'threshold': .5, **counts_metrics(matrix), **extras,
                  'provenance': 'Transcribed from the user-supplied Kaggle log on 2026-11-17',
                  'first_log_line': line}
        if source != 'custom_challenge':
            automatic = record['test_size'] - record['review_count']
            assert abs(automatic / record['test_size'] - record['automatic_coverage']) < 1e-12
            assert abs(1 - record['automatic_errors'] / automatic - record['automatic_accuracy']) < 1e-12
            assert record['urgent_automatically_found'] + record['urgent_automatically_missed'] + record['urgent_requiring_review'] == record['positive_count']
        results.append(record)
    dump('joint_minilm_20261117_results.json', results)
    definition = json.loads((ROOT / 'reports/final_minilm_trial/behavior_probe_definition.json').read_text())
    values = {
        'unresolved_upload': .8766919904553908, 'resolved_upload': .13179294484649998,
        'longer_resolution': .6148195379206594, 'problem_returned': .7857819888246814,
        'denied_access': .814495921703542, 'past_problem': .3473589755041708,
        'negative_statement': .5774006313339408, 'no_remaining_issue': .2276811886390194,
        'quoted_problem': .8161014841174796, 'reference_example': .1394147352825185,
        'still_waiting': .34602651229930526, 'different_failure': .5682769342136211,
        'thank_you': .024819751218269823, 'sharing_solution': .018711068722138416,
        'lost_work': .9069557122971804, 'received_feedback': .5787794723514542,
        'vague': .15940341968213023, 'missing_reference': .14781312289847318,
        'outside_weather': .5461510085673381, 'outside_recipe': .01148696547893856,
    }
    predictions = [{**row, 'score': values[row['id']], 'priority': int(values[row['id']] >= .5),
                    'action': 'NORMAL' if values[row['id']] <= .2 else
                    ('PRIORITY' if values[row['id']] >= .8 else 'REVIEW')}
                   for row in definition['probes']]
    labelled = [row for row in predictions if row['expected_binary_priority'] is not None]
    correct = sum(row['priority'] == row['expected_binary_priority'] for row in labelled)
    assert correct == 12 and len(labelled) == 16
    dump('joint_minilm_20261117_probes.json', {
        'provenance': 'Scores transcribed from log lines 1143–1322; texts match the frozen local probe definition.',
        'label_status': definition['label_status'], 'correct_binary_probes': correct,
        'diagnostic_binary_count': len(labelled), 'predictions': predictions,
        'binary_errors': [row['id'] for row in labelled if row['priority'] != row['expected_binary_priority']],
        'unlabelled_probes_automatically_classified': [row['id'] for row in predictions if row['expected_binary_priority'] is None and row['action'] != 'REVIEW'],
    })
    dump('joint_minilm_20261117_run.json', {
        'source': 'User-supplied complete Kaggle log; not local training or imported model weights',
        'notebook_url': 'https://www.kaggle.com/code/tmhiutv/minilm-confusion-final/notebook?scriptVersionId=370343895',
        'script_version_id': 370343895, 'mode': 'TRAIN', 'synthetic_training': False,
        'versions': {'python': '3.13.16', 'torch': '2.11.0+cu130', 'transformers': '5.20.2', 'sklearn': '1.6.1', 'sentencepiece': '0.2.2'},
        'device': 'cuda', 'visible_gpus': 2, 'training_gpu_count': 1,
        'total_log_seconds': 1008.5, 'archive_mb_decimal': 182.7,
        'data_checks_passed_in_log': True, 'count_metrics_independently_recomputed': True,
        'per_post_kaggle_csv_imported': False, 'kaggle_model_weights_imported': False,
        'best_epoch_by_seed': {'42': 3, '43': 3},
        'mean_validation_ap_by_epoch': {'42': [.6261234155523688, .701094181935056, .7270369449441203, .723499352022895],
                                        '43': [.6530186177023434, .7196721123913901, .7391745900583915, .7346162421569571]},
        'ten_epoch_trial_performed': False,
        'environment_limitation': 'The uploaded setup installed exact requested versions only when modules were absent; Transformers 5.20.2 remained installed despite a requested 4.57.6 pin.',
        'reproduction': 'Run this script to regenerate the transcribed records and count/probe checks; run the Kaggle notebook for actual fresh training, which may vary by environment.',
    })
    print('Recorded supplied Kaggle run: 4 real-test rows, 2 synthetic diagnostics, 20 probes; no training performed.')


if __name__ == '__main__':
    main()
