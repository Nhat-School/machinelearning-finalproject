"""Recompute required metrics from confusion counts in the user's supplied Kaggle log."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
# Transcribed from lines 1342-1365 of the Kaggle log supplied in this conversation.
COUNTS={
 ('upenn','upenn'):[[600,30,24,37],[602,28,29,32],[572,58,32,29],[630,0,61,0],[580,50,20,41]],
 ('stanford','stanford'):[[4352,357,280,825],[4257,452,232,873],[4126,583,301,804],[4709,0,1105,0],[4031,678,302,803]],
 ('upenn','stanford'):[[4425,284,698,407],[4502,207,750,355],[4215,494,728,377],[4709,0,1105,0],[4357,352,700,405]],
 ('stanford','upenn'):[[510,120,17,44],[500,130,15,46],[558,72,16,45],[630,0,61,0],[490,140,12,49]]}
MODELS=['tfidf_logreg','tfidf_svm','tfidf_nb','tfidf_dummy','minilm_logreg']
def main():
    results=[]
    for (source,target),matrices in COUNTS.items():
        for model,(tn,fp,fn,tp) in zip(MODELS,matrices):
            assert tn+fp+fn+tp=={'upenn':691,'stanford':5814}[target]
            results.append({'train_source':source,'test_source':target,'model':model,'confusion_matrix':[[tn,fp],[fn,tp]],
                'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn),
                'accuracy':(tn+tp)/(tn+fp+fn+tp),'test_size':tn+fp+fn+tp,'threshold':None,
                'provenance':'Confusion counts transcribed from user-supplied Kaggle log lines 1342-1365; metrics recomputed, not a local rerun.',
                'unavailable':'Per-post predictions, AP/AUC and exact thresholds were not imported for these baseline rows.'})
    (ROOT/'reports/kaggle_run/baseline_results_from_log.json').write_text(json.dumps(results,indent=2)+'\n')
    print('Recomputed P/R/F1/Accuracy for 20 supplied-log baseline confusion matrices.')
if __name__=='__main__':main()
