FINAL PROJECT FILES

FINAL_REPORT.docx
The official main report, copied without modifying its content.
Canonical source: ../reports/bao_cao_do_an_hoc_may.docx

FINAL_MODEL/
The saved default MiniLM model: real_only, seed 42, trained on UPenn and Stanford.
All files inside are required model assets. Keep their original internal names.
This is the saved local model, not newly trained Kaggle weights.

FINAL_KAGGLE.ipynb
Upload this notebook to Kaggle. Enable Internet and a GPU, then Run All.
It trains the two-real-dataset method and exports the seed 42 model.
It also runs the seed 43 control. New results may differ from the saved local run.
This copy has not been freshly trained on Kaggle during packaging.

RUN ON THIS MAC
Double-click FINAL_RUN_APP.command to open the existing project web app.
Double-click FINAL_RUN_MODEL.command to enter a post and use FINAL_MODEL directly.
Both launchers use the existing environment in the parent Project folder.
Keep FINAL inside Project; the launchers are not standalone installers.
The web app uses the canonical model in ../models; direct prediction uses FINAL_MODEL.

TXT FILES IN PROJECT
requirements-ml.txt: packages needed for machine learning and the web app.
requirements-ml.lock.txt: exact package versions recorded for reproducibility.
requirements-notebook.txt: packages for executing and validating notebooks locally.
requirements-reliability.txt: extra packages for the DeBERTa experiments.
requirements-report.txt: packages used to generate Word reports and images.
SHA256SUMS.txt: file fingerprints used to check integrity.
These are supporting files; keep them in their original locations.

The FINAL files are a snapshot. Regenerating the canonical report or retraining
models does not automatically update these copies.

Shell support scripts were moved to ../scripts/launchers/.
To rebuild from Project: bash scripts/launchers/reproduce.sh
To start from Project: bash scripts/launchers/start_app.sh
Older commands printed in the Word report use the former script locations.
