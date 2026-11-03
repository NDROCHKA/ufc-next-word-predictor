# Octagon Oracle — UFC Next-Word Prediction

Octagon Oracle completes an unfinished UFC sentence with two independently
trained PyTorch models: a bidirectional RNN and a TextCNN. React sends the input
over Socket.IO to Express, Express calls a local Python inference service, and
the interface shows both words so the user can choose one.

## What the system predicts

The models predict one **complete next word**. They do not complete partial
spellings and they do not predict future fight winners. Example prompts include:

- `khabib` → likely surname completion
- `conor mcgregor faced` → likely opponent first name
- `islam makhachev competed in a` → likely division
- `the referee can stop the` → likely UFC vocabulary

The corpus focuses on UFC bouts dated 2018–2025. It includes the 80 fighters
with the most appearances in that period plus a checked set of featured fighters
when those names occur in the source data.

## Architecture

```text
React UI
  └─ Socket.IO: prediction:request
       └─ Express gateway
            └─ HTTP POST /predict
                 └─ Python / FastAPI
                      ├─ BiRNN checkpoint
                      └─ TextCNN checkpoint
       └─ Socket.IO: prediction:result
            └─ user chooses either word
```

Responsibilities are intentionally separated:

- `ml/src/ufc_next_word/data`: validation, corpus generation, tokenization, and vocabulary
- `ml/src/ufc_next_word/models`: BiRNN and TextCNN architectures
- `ml/src/ufc_next_word/training`: training, early stopping, and metrics
- `ml/src/ufc_next_word/inference`: checkpoint loading and prediction use case
- `ml/src/ufc_next_word/api`: FastAPI transport adapter
- `server/services`: Express adapter for the Python service
- `server/socket`: Socket.IO input validation and event handling
- `client/src`: React presentation and user interaction

## Dataset provenance

The raw snapshot is `data/raw/ufcstats_2013_2025.csv`, downloaded from the
public repository
[`01x3ATM3-UFC_FIGHT_DATA_2013_TO_2025`](https://github.com/01x3ATM3/01x3ATM3-UFC_FIGHT_DATA_2013_TO_2025).
That repository states that it scraped UFCStats on May 26, 2025. Each generated
sentence retains the original UFCStats fight URL and generated fight ID.

This project does not treat the mirror as an official UFC publication. Corpus
generation checks dates, required fields, and W/L outcome columns. Templates
word verified fields into sentences; they do not invent matchups or winners.

All sentence variants from a bout remain in one split. This prevents the same
fact, rephrased by another template, from appearing in both training and test
data. The vocabulary is built from the training split only.

## One-time setup

Install Node dependencies if they are not already present:

```powershell
npm.cmd install
npm.cmd --prefix client install
```

From the `ml` folder, create the Python environment with uv:

```powershell
$env:UV_CACHE_DIR = "..\.uv-cache"
uv sync --extra dev --python 3.11
```

Plain `python -m venv` plus `pip install -e ".[dev]"` also works when Python
3.11+ is already installed.

## Prepare, train, and inspect

Run these commands from `ml`:

```powershell
uv run python scripts/prepare_data.py
uv run python scripts/train.py
uv run python scripts/predict.py "khabib"
uv run pytest
```

Training writes these submission artifacts:

- `artifacts/birnn.pt`
- `artifacts/textcnn.pt`
- `artifacts/vocabulary.json`
- `artifacts/model_config.json`
- `artifacts/training_report.json`

`training_report.json` contains the complete per-epoch history plus final test
top-1 and top-3 accuracy. The test set is evaluated only after selecting the
best validation checkpoint.

### Measured run included in this repository

The saved checkpoints were trained on CPU with seed 433, a maximum of 12 epochs,
batch size 128, and early-stopping patience 3.

| Model | Best validation loss | Test top-1 | Test top-3 | Training time |
|---|---:|---:|---:|---:|
| BiRNN | 1.3996 | 67.58% | 77.23% | 108.22 s |
| TextCNN | 1.7197 | 65.50% | 75.58% | 72.02 s |

The vocabulary contains 1,212 words. Training produced 73,320 prefix examples;
validation used 8,441 and the untouched test set used 7,957. These figures come
from `artifacts/training_report.json`, not from hand-entered expected results.

## Run the application

Open three terminals.

Terminal 1 — Python model service, from `ml`:

```powershell
uv run uvicorn ufc_next_word.api.app:app --host 127.0.0.1 --port 8000
```

Terminal 2 — Express and Socket.IO, from the repository root:

```powershell
npm.cmd start
```

Terminal 3 — React, from `client`:

```powershell
npm.cmd run dev
```

Open `http://localhost:5173`. The status pill becomes `connected` when React has
reached Express. Express health is available at `http://localhost:3000/api/health`
and Python health at `http://127.0.0.1:8000/health`.

## Evaluation interpretation

Top-1 accuracy requires the model's first word to match the recorded target.
Top-3 accuracy accepts the target anywhere among the first three suggestions.
Natural language can have several valid next words, so these metrics are useful
comparisons between the models rather than a complete measure of sentence quality.

The percentage shown in the UI is the model's softmax score. It is not a
calibrated probability or a guarantee that the word is correct.

## Verification

```powershell
npm.cmd test
npm.cmd --prefix client run build
```

Python tests check tokenization, fact-to-sentence outcome handling, deterministic
splitting, and output shapes for both neural networks.

