"""Keep canonical datasets together and preserve existing pipeline paths with relative links."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'Datasets'
SOURCES={'UPenn':('upenn','All_Courses_REDACTED_CODED.csv'),
         'Stanford':('stanford','Stanford.csv')}

def main():
    OUT.mkdir(exist_ok=True);audit=json.loads((ROOT/'reports/data_audit.json').read_text());catalog={}
    for name,(source,raw_name) in SOURCES.items():
        directory=OUT/name;directory.mkdir(exist_ok=True)
        for legacy,target in [(ROOT/'data/raw'/raw_name,directory/'raw.csv'),
                              (ROOT/f'data/processed/{source}.csv',directory/'processed.csv')]:
            if legacy.is_symlink():
                assert legacy.resolve()==target.resolve() and target.exists()
                continue
            assert legacy.is_file(),f'Prepare data first: {legacy}'
            before=hashlib.sha256(legacy.read_bytes()).hexdigest()
            if target.exists():
                assert hashlib.sha256(target.read_bytes()).hexdigest()==before
                legacy.unlink()
            else:legacy.replace(target)
            legacy.symlink_to(os.path.relpath(target,legacy.parent))
            assert hashlib.sha256(legacy.read_bytes()).hexdigest()==before
        metadata={'name':name,'source_key':source,'provenance':'original research dataset',
                  'raw_sha256':audit['datasets'][source]['raw_sha256'],
                  'processed_sha256':hashlib.sha256((directory/'processed.csv').read_bytes()).hexdigest(),
                  'clean_posts':audit['datasets'][source]['counts']['final_rows'],
                  'original_labels_preserved':True,'binary_label':'priority = int(urgency > 4)',
                  'source_url':audit['datasets'][source]['url']}
        (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        (directory/'README.md').write_text(f'# {name}\n\nraw.csv is the original pinned CSV. processed.csv is the audited cleaned version.\n'
            'Original urgency labels and fixed train/test assignments are preserved. Dataset metadata and source URL are in metadata.json.\n'
            'Legacy paths under Project/data are relative symbolic links to these canonical files.\n')
        catalog[name]=metadata
    custom=OUT/'CustomForum';custom.mkdir(exist_ok=True)
    catalog['CustomForum']={'provenance':'assistant-authored synthetic scenarios; not real student records',
                           'role':'reviewable development/challenge data',
                           'training_policy':'Only approved, adjudicated training rows may be included in training.',
                           'status':'Draft labels require human review; no independent accuracy claim.'}
    (OUT/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
    (OUT/'README.md').write_text('# Project datasets\n\n'
        '- UPenn: original and processed source dataset.\n- Stanford: original and processed source dataset.\n'
        '- CustomForum: clearly marked synthetic scenarios, annotation guidance and review fields.\n\n'
        'The two real datasets are not expanded by disguising generated examples as original posts.\n'
        'Custom data stays separate. All related scenario variants must remain in the same split.\n'
        'Run python Project/scripts/organize_datasets.py after data preparation to restore this layout.\n')
    print('Canonical dataset folder:',OUT)
    print('Original file hashes verified; legacy training paths remain usable.')

if __name__=='__main__':main()
