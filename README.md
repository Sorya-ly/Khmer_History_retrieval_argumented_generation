# Khmer History RAG: Part 1, 
**Course:** ITM454 | Natural Language Processing
## Preprocessing and Splitting

**Owner:** LY Soryapheak | **Due:** October 3, 2026
**Files:** `Preprocessing_splitting.ipynb` (pipeline), `khmer_history_chunks.json` (output)

## 1. Purpose

This stage turns raw Khmer Wikipedia articles into clean, sentence-aware text chunks with metadata. The output is the input for the next stages:

| Part | Task | Due |
|------|------|-----|
| **1 LY Soryapheak** | Preprocessing + splitting | Oct 3, 2026 |
| 2 | Embedding + fine-tuning + storing | Oct 24, 2026 |
| 3 | Evaluation + retriever + LLM | Oct 30, 2026 |

## 2. Data source

- **Source:** Khmer Wikipedia (`km.wikipedia.org`), fetched through the official API using the `wikipedia-api` library.
- **Page(s):** ប្រវត្តិសាស្ត្រកម្ពុជា (History of Cambodia) [add any further pages here].
- **Fetched on:** [date].

## 3. Pipeline

```
Wikipedia API -> section walk -> cleaning -> sentence split -> chunking -> JSON
```

1. **Fetching.** Pulls the article intro plus all sections and nested subsections, and records each section path (e.g. `សម័យ > ក្រោម`). Sections without historical content (References, Notes, External links, See also, Bibliography) are skipped.
2. **Cleaning.**
   - Unicode NFC normalization.
   - Removal of zero-width characters.
   - Removal of citation markers (`[1]`), heading markup (`==`, `**`), HTML tags and `{{templates}}`.
   - Whitespace collapse.
3. **Sentence splitting.** Uses `khmernltk.sentence_tokenize`. Any oversized "sentence" is cut on word boundaries as a safety net.
4. **Chunking.** A sentence-aware sliding window. Sizes are measured in real Khmer words with `khmernltk.word_tokenize`. Chunks never cut a sentence in half, and each chunk repeats the last sentences of the previous one for context.

## 4. Settings

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `chunk_size` | 180 words | Maximum words per chunk |
| `overlap_words` | 30 words | Words repeated from the previous chunk |
| `min_words` | 40 words | A shorter final chunk is merged into the previous one |

## 5. Output format

`khmer_history_chunks.json` is a list of objects:

```json
{
  "chunk_id": "km-hist-003-01",
  "text_payload": "clean chunk text",
  "text_for_embedding": "Section heading: clean chunk text",
  "metadata": {
    "source_url": "https://km.wikipedia.org/wiki/...",
    "document_domain": "Khmer Wikipedia API",
    "page_title": "ប្រវត្តិសាស្ត្រកម្ពុជា",
    "historical_era_heading": "Parent > Child section",
    "section_index": 3,
    "chunk_index": 1,
    "n_words": 176,
    "n_chars": 812
  }
}
```

**Notes for Part 2 (embedding):**
- Embed `text_for_embedding`. It includes the section heading, which gives the model era context.
- Keep `metadata` and `chunk_id` alongside each vector so answers can cite `source_url` and the era heading.

## 6. Results

| Metric | Value |
|--------|-------|
| Pages fetched | [n] |
| Sections/records kept | [n] |
| Total chunks | [n] |
| Words per chunk (min / avg / max) | [min] / [avg] / [max] |

## 7. How to run

```bash
pip install khmer-nltk wikipedia-api
```

Then run the notebook cells in order. The final cell writes `khmer_history_chunks.json`.

Note that the package installs as `khmer-nltk`, but you import it as `khmernltk`. Likewise `wikipedia-api` is imported as `wikipediaapi`.

## 8. Known limitations

- Word counts depend on the `khmernltk` CRF tokenizer, which can make mistakes on names and rare words.
- Only [n] Wikipedia page(s) are used, so the knowledge base is limited to what those pages cover.
- Wikipedia content may contain errors or bias, and its text can change over time.
- Section skipping uses a fixed heading list, so pages with different heading names may need it updated.
- Tables, infobox contents and image captions are not included.

## 9. Dependencies

`wikipedia-api`, `khmer-nltk`, `python-crfsuite`, `scikit-learn`, Python 3.12.
