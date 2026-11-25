# Khmer History RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions about
Cambodian history **in Khmer**, grounded in 908 passages from 41 Khmer Wikipedia
articles.

You ask a question in Khmer. The system finds the most relevant passages and
Gemini writes an answer using only those passages, with the sources shown.

```
question (Khmer)
  -> fine-tuned multilingual-e5-small   ("query: " prefix)
  -> hybrid score: 0.5 * dense + 0.5 * BM25      -> top 50
  -> cross-encoder rerank (bge-reranker-v2-m3)   -> top 5
  -> Gemini 2.5 Flash, answering only from those 5 passages
  -> Gradio chat UI
```

---

## Results

Measured on **59 held-out test questions**. Each question has exactly one
correct passage among all 908. The questions are **reworded** so they avoid the
passage's own wording, which is closer to how real users ask.

| System | Recall@1 | Recall@5 |
|---|---|---|
| Fine-tuned e5 alone | 20/59 (0.339) | 40/59 (0.678) |
| Hybrid (BM25 + fine-tuned e5) | 23/59 (0.390) | 42/59 (0.712) |
| **Hybrid → reranker (top 50 → top 5)** | **37/59 (0.627)** | **53/59 (0.898)** |

Retrievers compared on their own (Recall@1, original vs reworded questions):

| System | Original | Reworded |
|---|---|---|
| BM25 (lexical) | **0.814** | 0.288 |
| e5-small zero-shot | 0.508 | 0.136 |
| e5-small fine-tuned | 0.576 | 0.322 |
| Hybrid (BM25 + fine-tuned e5) | 0.763 | **0.424** |

BM25 wins when the question reuses the passage's words but fails when it is
reworded. Fine-tuning makes e5 more robust to rewording, the hybrid combines
both, and the reranker gives the largest single gain.

The full analysis (examples, ablations, data imbalance, confidence intervals,
limitations) is in [`RESULTS.md`](RESULTS.md).

---

## Repository structure

```
.
├── src/
│   ├── Preprocessing_splitting.ipynb      # 1. scrape, clean, chunk, train/test split
│   ├── embedding_finetune_908chunks.ipynb # 2. embed, fine-tune e5, build the store
│   ├── evaluate_retriever.ipynb           # 3. compare BM25 / e5 / hybrid
│   └── rag_chat.ipynb                     # 4. retriever + reranker + Gemini + Gradio
├── data/
│   ├── khmer_history_chunks.json          # all 908 chunks with source metadata
│   ├── khmer_history_train.json           # train-split chunks
│   ├── khmer_history_test.json            # test-split chunks
│   ├── rag_store.json                     # chunks + embedding settings for the chatbot
│   ├── gold.csv                           # 164 question → chunk_id pairs
│   └── gold_test_paraphrased.csv          # 59 reworded test questions
├── khmer-retriever-finetuned/             # fine-tuned e5 (config + tokenizer; weights separate)
├── emb_*.npy                              # precomputed embeddings (e5, bge-m3, fine-tuned, final)
├── finetune_demo_numpy.py                 # tiny NumPy demo of the contrastive fine-tuning loss
├── RESULTS.md                             # full results and analysis
├── SETUP.md                               # detailed setup notes
└── requirements.txt
```

---

## Pipeline

### 1. Preprocessing and splitting
- Scrapes 41 Khmer Wikipedia history articles section by section, skipping
  non-content sections (references, external links).
- Cleans the text: Unicode NFC normalisation, zero-width character removal,
  citation/template/HTML removal.
- Splits each section into sentence-aware chunks of about 180 Khmer words with a
  30-word overlap, giving **908 chunks from 379 sections**.
- Splits at the **chunk level** into train and test so test questions are about
  passages the retriever never saw during fine-tuning.

### 2. Embedding, fine-tuning and store
- Base model: [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small),
  compared against `BAAI/bge-m3` as a zero-shot reference.
- Fine-tuned on question–passage pairs with `MultipleNegativesRankingLoss`
  (3 epochs, 10% warmup), using the e5 `query:` / `passage:` prefixes.
- Chunks are indexed as `historical_era_heading + ": " + text_payload`.
- Unit-normalised embeddings are saved so a single dot product scores every chunk.

### 3. Evaluation, retriever and LLM
- BM25 with Khmer word segmentation (`khmer-nltk`), stopword removal and
  bigrams (bigrams were the largest measured BM25 gain).
- Hybrid retrieval: `0.5 * dense + 0.5 * BM25`.
- Reranking with `BAAI/bge-reranker-v2-m3` (not fine-tuned).
- Generation with Gemini 2.5 Flash, which sees only `text_payload` and is told
  to answer from the retrieved passages alone.
- A relevance gate (`RERANK_THRESHOLD`) politely refuses off-topic questions.
- Metrics: Recall@1, Recall@5, MRR@10 on original and reworded test questions.

---

## Getting started

### 1. Get the model weights
`khmer-retriever-finetuned/model.safetensors` (470 MB) is too large for GitHub
and is not in this repo. Copy it from the team's shared Drive into
`khmer-retriever-finetuned/`.

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# NVIDIA GPU only: install the CUDA build of PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cu121
# Mac (Apple Silicon) or CPU: plain `pip install torch` is enough

pip install -r requirements.txt
```

### 3. Add your Gemini API key

```bash
cp .env.example .env
# then put your key in .env:  GEMINI_API_KEY=...
```

Get a key at <https://aistudio.google.com/app/apikey>. Without a key, retrieval
still works and the app shows the retrieved passages instead of an answer.

### 4. Run the chatbot

```bash
jupyter notebook src/rag_chat.ipynb
```

Run the cells in order; the last cell launches the Gradio chat. The first run
downloads the reranker (about 2.3 GB) from Hugging Face.

See [`SETUP.md`](SETUP.md) for common pitfalls. In short: keep the e5 prefixes,
and **do not regenerate the chunks**, since that renumbers every `chunk_id` and
invalidates the gold questions and the fine-tuned model's training data.

---

## Limitations

- **Small dataset:** 164 labelled questions; only 65 of 908 chunks have a
  question, and the 95% confidence intervals are about ±0.12 wide.
- **Uneven coverage:** about half of all year mentions are from the 1900s, while
  the French colonial period has only 2 chunks.
- **Inconsistent spelling** of names in Khmer Wikipedia (e.g. ជ័យវរ្ម័ន / ជយវម៌្ម)
  and mixed Khmer/Arabic digits hurt keyword search.
- **Source reliability:** answers are only as accurate as Khmer Wikipedia.

---

## Team

| Date | Stage | Member |
|---|---|---|
| October 3, 2026 | Preprocessing + splitting | Soryapheak LY |
| October 24, 2026 | Embedding + fine-tuning + store | Sophaltheany SEAN |
| October 30, 2026 | Evaluation + building the retriever + LLM | Houykea LIM |

---

## Acknowledgements

- Text from [Khmer Wikipedia](https://km.wikipedia.org), licensed CC BY-SA 4.0.
- Models: [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small),
  [bge-m3](https://huggingface.co/BAAI/bge-m3),
  [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3),
  Google Gemini 2.5 Flash.
- Khmer word segmentation: [khmer-nltk](https://github.com/VietHoang1512/khmer-nltk).
