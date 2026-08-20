# ClearScan Use Case

This folder contains a minimal "usecase" repository extracted from the ClearScan project for sharing and demonstration.

Contents:

- `README.md` — this file

How to use

1. Copy or symlink the files you need from the main project (e.g. `streamlit_app.py`, `run_full_pipeline.py`) into this folder.
2. Create and activate a Python virtual environment and install dependencies from the main project's `requirements.txt`.

Example commands (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ..\requirements.txt
streamlit run streamlit_app.py
```

If you want me to add a complete runnable subset (Streamlit app + run script) into this repo, tell me which files to include and I'll copy them here and commit.