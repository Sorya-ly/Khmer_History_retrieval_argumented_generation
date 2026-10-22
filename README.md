# Khmer History RAG

**Course:** ITM454 | **Due:** October 30, 2026

**Files:** `Preprocessing_splitting.ipynb` (pipeline), `khmer_history_chunks.json` (all chunks), `khmer_history_train.json` / `khmer_history_test.json` (train/test split)

## 1. Purpose

This stage turns raw Khmer Wikipedia articles into clean, sentence-aware text chunks with metadata. The output is the input for the next stages:

| Part | Task | Due |
|------|------|-----|
| **1 LY Soryapheak** | Preprocessing + splitting | Oct 3, 2026 |
| **2 SEAN Sophaltheany**| Embedding + fine-tuning + storing | Oct 24, 2026 |
| **3 LIM Houykea** | Evaluation + retriever + LLM | Oct 30, 2026 |

## 2. Data source

- **Source:** Khmer Wikipedia (`km.wikipedia.org`), fetched through the official API using the `wikipedia-api` library.
- **Pages:** 41 articles, grouped by period (chunk counts in brackets):
  - **Overview / pre-Angkor / Angkor (9):** ប្រវត្តិសាស្ត្រកម្ពុជា (78), ចេនឡា (42), អាណាចក្រខ្មែរ (67), ដើមសម័យអង្គរ (7), ចុងសម័យអង្គរ (33), សម័យមហានគរ (14), ជ័យវម៌្មទី ២ (7), សូរ្យវរ្ម័នទី២ (19), ជ័យវរ្ម័នទី៧ (26)
  - **Post-Angkor / colonial (5):** យុគអន្ធកាលនៃកម្ពុជា (25), លង្វែក (13), សម័យកាលលង្វែក (10), សមរភូមិលង្វែក (១៥៩៤) (4), សម័យអាណានិគមនិយមបារាំង (2)
  - **Modern (14):** នរោត្ដម សីហនុ (46), សាធារណរដ្ឋខ្មែរ (28), លន់ នល់ (21), ខ្មែរក្រហម (10), កម្ពុជាប្រជាធិបតេយ្យ (28), ការគ្រប់គ្រងរបស់ខ្មែរក្រហមនៅកម្ពុជា (73), ប្រវត្តិនៃរបបខ្មែរក្រហម (31), សាធារណរដ្ឋប្រជាមានិតកម្ពុជា (73), រដ្ឋាភិបាលចម្រុះកម្ពុជាប្រជាធិបតេយ្យ (8), អាជ្ញាធរអន្តរកាលរបស់អង្គការសហប្រជាជាតិនៅកម្ពុជា (13), សាលាក្ដីខ្មែរក្រហម (8), នរោត្តម សីហមុនី (27), រាជានិយមកម្ពុជា (28), កាលប្បវត្តិកម្ពុជា (10)
  - **Temples (13):** ប្រាសាទ អង្គរវត្ត (15), ប្រាសាទសម្បូរព្រៃគុហ៍ (11), ប្រាសាទនាគព័ន្ធ (7), ប្រាសាទបាយ័ន (15), ប្រាសាទតាព្រហ្ម (6), ប្រាសាទបន្ទាយស្រី (17), ប្រាសាទភ្នំបាខែង (8), ប្រាសាទព្រះខ័ន (13), ប្រាសាទបាពួន (11), ប្រាសាទតាកែវ (8), ប្រាសាទកោះកេរ (9), ប្រាសាទភ្នំជីសូរ (22), ប្រាសាទភ្នំដា (15)
- **Fetched on:** October 22, 2026.

## 3. Pipeline

```
Wikipedia API -> section walk -> cleaning -> sentence split -> chunking -> JSON -> train/test split
```

1. **Fetching.** Pulls the article intro plus all sections and nested subsections, and records each section path (e.g. `សម័យ > ក្រោម`). Sections without historical content (References, Notes, External links, See also, Bibliography) are skipped.
2. **Cleaning.**
   - Unicode NFC normalization.
   - Removal of zero-width characters.
   - Removal of citation markers (`[1]`), heading markup (`==`, `**`), HTML tags and `{{templates}}`.
   - Whitespace collapse.
3. **Sentence splitting.** Uses `khmernltk.sentence_tokenize`. Any oversized "sentence" is cut on word boundaries as a safety net.
4. **Chunking.** A sentence-aware sliding window. Sizes are measured in real Khmer words with `khmernltk.word_tokenize`. Chunks never cut a sentence in half, and each chunk repeats the last sentences of the previous one for context.
5. **Train/test split.** Chunks under 30 words and chunks from skipped headings are dropped. The remaining chunks are grouped by section, and whole sections are assigned to the test set until it holds about 20% of the chunks (seed 42). Because the split is done by section, overlapping chunks from the same section never end up on both sides. The notebook asserts that no `chunk_id` appears in both sets.

## 4. Settings

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `chunk_size` | 180 words | Maximum words per chunk |
| `overlap_words` | 30 words | Words repeated from the previous chunk |
| `min_words` | 40 words | A shorter final chunk is merged into the previous one |
| `TEST_RATIO` | 0.2 | Target share of chunks in the test set |
| `MIN_WORDS` (split) | 30 words | Chunks shorter than this are left out of train/test |
| `SEED` | 42 | Random seed for the section shuffle |

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
| Pages fetched | 41 |
| Sections with at least one chunk | 379 |
| Total chunks | 908 |
| Words per chunk (min / avg / max) | 3 / 139.7 / 272 |
| Usable chunks after filtering | 867 (41 dropped) |
| Train / test chunks | 694 / 173 (80% / 20%) |

## 7. How to run

```bash
pip install khmer-nltk wikipedia-api
```

Then run the notebook cells in order. The pipeline cell writes `khmer_history_chunks.json`, and the split cell after it writes `khmer_history_train.json` and `khmer_history_test.json`.

Note that the package installs as `khmer-nltk`, but you import it as `khmernltk`. Likewise `wikipedia-api` is imported as `wikipediaapi`.

## 8. Known limitations

- Word counts depend on the `khmernltk` CRF tokenizer, which can make mistakes on names and rare words.
- Only 41 Wikipedia pages are used, so the knowledge base is limited to what those pages cover. Some pages are very short (e.g. សម័យអាណានិគមនិយមបារាំង has only 2 chunks), so the colonial period is thinly covered.
- Short sections can still produce very small chunks (as few as 3 words), because `min_words` only merges a short tail chunk within a section.
- Wikipedia content may contain errors or bias, and its text can change over time.
- Section skipping uses a fixed heading list, so pages with different heading names may need it updated.
- Tables, infobox contents and image captions are not included.

## 9. Dependencies

`wikipedia-api`, `khmer-nltk`, `python-crfsuite`, `scikit-learn`, Python 3.12.
