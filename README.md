# ML-Based Identity Threat Detection and Response (ITDR) — Prototype

Working prototype of the framework in *"Machine Learning-Based Identity Threat Detection and Response"*.
Python backend + **Streamlit** front end. All data is synthetic.

## Run in VS Code
1. Install Python 3.10+ and the VS Code **Python** extension.
2. `File > Open Folder...` and choose this folder.
3. Open a terminal (`Ctrl+` `` ` ``) and run:
   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. In VS Code press `Ctrl+Shift+P` > **Python: Select Interpreter** > pick `.venv`.
5. Your browser opens at http://localhost:8501.

## How it maps to the article
| Article stage | File |
|---|---|
| Identity sources (IdP, MFA, PAM, cloud audit) | `itdr/data_gen.py` (synthetic; replace with real log ingestion) |
| Normalisation + feature engineering | `itdr/features.py` |
| Rules + anomaly (Isolation Forest) + supervised (Logistic Regression) → risk tier | `itdr/detection.py` |
| Explainable alerts (observed evidence vs. model indicators) | `itdr/explain.py` |
| Policy-based response, approval, dry-run, audit | `itdr/response.py` |
| Evaluation plan (baselines i–iv, P/R/F1/FPR/PR-AUC, time-based split) | `itdr/detection.py::evaluate` |
| Analyst UI | `app.py` |

## Design choices that follow the article
- ML supports decisions; it never acts alone. High-impact actions need corroboration + approval.
- Execution is **dry-run** by default; `execute()` is the one place to connect a real IdP.
- Analyst feedback is logged, not auto-retrained.
- Explanations separate observed facts from model-derived indicators.
- No performance claims: the numbers shown come from synthetic data and a small test set.

## Next steps
Replace `data_gen.py` with a loader for real/de-identified logs (or the CERT dataset), add detection-latency and
ablation experiments, try tree models with `shap`, and connect `execute()` to an identity provider.
