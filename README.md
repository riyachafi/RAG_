# A Practical Guide to RAG – Retrieval-Augmented Generation over PDFs

Python · PyMuPDF · Sentence-Transformers · FAISS · Hugging Face Transformers · Qwen2.5-Instruct

## Project structure
```
rag_lab/
├── config.py              # all settings (paths, models, thresholds, the 4 experiments)
├── rag_pipeline.py        # core library: load -> clean -> chunk -> embed -> FAISS -> retrieve -> generate
├── build_index.py         # build and save an index
├── ask.py                 # question answering (CLI, shows file + page)
├── evaluate.py            # Precision@K, Recall@K, MRR, correctness, relevance, grounding + experiments
├── make_sample_pdfs.py    # creates sample PDFs for testing
├── requirements.txt
├── README.md
├── data/
│   ├── documents/         # put your PDFs here
│   ├── index/             # one sub-folder per experiment (index.faiss, chunks.json, index_info.json)
│   └── eval_dataset.json  # evaluation questions + reference answers
└── results/               # per-experiment details + experiment_comparison.{csv,json,md,png}
```

## Installation on Windows (PowerShell)
1. Install **Python 3.10 or 3.11** (tick *Add Python to PATH*).
2. Open PowerShell in the project folder:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
# CPU-only PyTorch (smaller download). Skip this line if you have an NVIDIA GPU and want CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
3. Choose the LLM (the first run downloads it from Hugging Face; 3B ≈ 6 GB, 1.5B ≈ 3 GB):
```powershell
# Light model for CPU / < 16 GB RAM (recommended on most laptops):
$env:RAG_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Full model (needs a GPU with >= 8 GB VRAM or >= 16 GB RAM):
$env:RAG_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
```
(In `cmd.exe` use `set RAG_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct`.)

## Run
```powershell
python make_sample_pdfs.py                      # optional: sample PDFs (or copy your own into data\documents)
python build_index.py                           # baseline index (chunk 900, overlap 150)
python ask.py --question "What is chunk overlap used for?"
python ask.py                                   # interactive mode
python ask.py --retrieval-only --question "What is FAISS?"   # no LLM needed
python evaluate.py                              # retrieval metrics for the 4 experiments
python evaluate.py --with-generation            # + generation metrics (loads the LLM)
```
Other experiments: `python build_index.py --experiment variant_a` (also `variant_b`, `variant_c`) and `python ask.py --experiment variant_a`.

## Behaviour to know
* The answer always shows **source file and page number**.
* If the documents do not contain the answer, the system prints **"The answer is not available in the provided documents."** (similarity gate, `NOT_FOUND` from the model, or grounding check).
* **Scanned PDFs** have no text layer: run OCR first (`ocrmypdf in.pdf out.pdf`), then re-build the index.
* Tune `RAG_MIN_SCORE` (default 0.25) and `RAG_GROUNDING_MIN` (default 0.30) for your documents.
* Reproducibility: seed 42, greedy decoding, exact FAISS search; settings are saved in `results/experiment_comparison.json`.

## Common errors
See section 10 of the notebook (missing modules, CUDA/CPU memory, model download, always "not available", Windows encoding).
