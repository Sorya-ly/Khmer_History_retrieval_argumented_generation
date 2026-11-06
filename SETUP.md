# Setup

Clone the repo, then do these three things before running anything.

## 1. Get the model weights

`khmer-retriever-finetuned/model.safetensors` is **not in this repo**. It is
470 MB, over GitHub's 100 MB file limit, so it is git-ignored.

Copy it from the shared Drive folder:

```
khmer_rag/khmer-retriever-finetuned/model.safetensors
        ->  khmer-retriever-finetuned/model.safetensors
```

Everything else in that folder (config, tokenizer, pooling) is already here.
Without this one file `SentenceTransformer` fails with a confusing error.

If the model is later pushed to Hugging Face, set `MODEL_PATH` in
`src/rag_chat.ipynb` to the repo id instead and skip this step.

## 2. Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU build
pip install -r requirements.txt
```

Order matters. Letting step 2 pull torch from PyPI gives the CPU-only build,
and the reranker then takes several seconds per question instead of about two.

## 3. Add your Gemini key

```bash
cp .env.example .env
```

Put a real key in it. The file is git-ignored. Without it retrieval still runs
and the app shows the retrieved passages instead of a generated answer.

---

# Running

```bash
jupyter notebook src/rag_chat.ipynb
```

Run the cells in order. The last one launches the Gradio chat.

First run downloads the reranker, about 2.3 GB, from Hugging Face. It is not
stored in this repo or in Drive.

---

# The pipeline

```
question (Khmer)
  -> e5 fine-tuned encode, "query: " prefix
  -> hybrid: 0.5 * dense + 0.5 * BM25   ->  top 50
  -> cross-encoder rerank               ->  top 5
  -> Gemini 2.5 Flash, grounded in those 5
```

Measured on the 59 held-out reworded test questions, where the question
deliberately avoids the passage's own wording:

| System | Recall@1 | Recall@5 |
|---|---|---|
| fine-tuned e5 alone | 20 / 59 | 40 / 59 |
| hybrid (BM25 + e5) | 23 / 59 | 42 / 59 |
| hybrid -> rerank -> top 5 | **37 / 59** | **53 / 59** |

The reranker is not fine-tuned and needed no training data.

---

# Things that will bite you

**Passage text.** Chunks are indexed as
`historical_era_heading + ": " + text_payload`, because that is what the
retriever was fine-tuned on and what the scores above were measured with. The
language model is shown `text_payload` alone, so no heading leaks into answers.
Changing either breaks the comparison with the reported numbers.

**The e5 prefixes.** `query: ` on questions, `passage: ` on chunks. Omit them
and nothing errors; retrieval just quietly gets worse. The reranker takes the
raw question with no prefix.

**Do not regenerate the chunks.** Re-running Part 1 renumbers every
`chunk_id`, which invalidates all 164 gold questions, the paraphrased test set,
the fine-tuned model's training data, and any citation already produced.

**The relevance gate.** `RERANK_THRESHOLD` in the generation cell decides when
a question is off-topic and gets a polite refusal instead of an answer. It is
set to 0.30 as a starting point. Calibrate it: print `rerank_score` for a few
real history questions and a few greetings, then pick a value between them.
