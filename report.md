# RAG: Baseline vs. Re-Ranking — Report

**Repository:** https://github.com/amitabh-anandpd/rag-biencoder-vs-crossencoder
**Notebook:** `rag_baseline_vs_rerank.ipynb`

## Setup

| Component | Choice |
|---|---|
| Document | GDPR, Regulation (EU) 2016/679, 88 pages |
| Chunking | 800-character windows, 150 overlap, per page → 580 chunks |
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` (384-d), FAISS `IndexFlatIP` on normalised vectors (cosine) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `gemini-3.5-flash-lite`, temperature 0, same prompt for both pipelines |
| Pipelines | P1: bi-encoder top 10 → top 3 to LLM. P2: same top 10 → cross-encoder re-sort → top 3 to LLM |

---

## 1. Architecture Comparison

✍️ *Write 1–2 short paragraphs in your own words.*

Notes:
- Bi-encoder: query and chunk encoded **separately** → one vector each → cosine similarity.
- Chunk vectors are computed once, offline; a query costs one encoding plus an index search.
- Whole chunk squeezed into one vector → small qualifiers ("not", "except", "shall not be required") barely move it; topic overlap dominates.
- Cross-encoder: query and chunk fed **together** (`[CLS] query [SEP] chunk [SEP]`); self-attention connects every query token with every chunk token.
- Outputs one relevance score (a logit, so only the order matters); nothing can be precomputed.
- Both models are 6-layer MiniLM → the difference comes from the architecture, not size.

---

## 2. Qualitative & Rank Shift Analysis

### Rank shifts of the key chunk

| # | Failure type | Query | Key chunk | Bi-encoder rank | Cross-encoder rank |
|---|---|---|---|---|---|
| 1 | Keyword trap | Can my employer monitor me at work? | 562 (p.84, Art. 88) | #3 | **#1** |
| 2 | Exception | Can I get my personal data deleted, and when can the company refuse? | 99 (p.13, Recital 65) | #6 | **#1** |
| 3 | Negation | When does a company not need to tell affected people about a breach? | 378 (p.53, Art. 34(3)) | #6 | **#2** |
| 4 | Buried chunk | Must every company appoint a data protection officer? | 392 (p.55, Art. 37(1)(c)) | #7 | **#1** |
| 5 | Control | Can I take my data to another service provider? | 102 (p.13, Recital 68) | #2 | #1 |

### Answer 1 vs. Answer 2

| # | Answer 1 (baseline) | Answer 2 (re-ranked) | Failure in Answer 1 |
|---|---|---|---|
| 1 | Correct (Art. 88 chunk was at #3) | Correct, adds collective agreements (p.29) | None, but 2 of 3 chunks were distractors |
| 2 | Lists erasure grounds only, nothing on refusal | Lists all 5 refusal grounds | Omitted essential clause |
| 3 | "Context does not list exceptions" | Gives encryption + later-measures exceptions | Omitted essential clause |
| 4 | "May designate… shall where required by law" | Gives mandatory large-scale condition | Misstated rule |
| 5 | Correct | Correct | None (control) |

✍️ *Write a short paragraph per query (or one combined paragraph) describing what moved and why it mattered.*

Notes:
- Q1: baseline #1 and #2 (chunks 195, 196) are about supervisory authorities' "monitoring and enforcement", a lexical match but the wrong topic. Bi-encoder scores for all 10 sit between 0.315 and 0.367; the cross-encoder spread them from −5.8 to −11.2.
- Q2: baseline top 3 were all about when you *can* erase (Art. 17(1), start of Recital 65).
- Q3: baseline pulled Art. 33 (notifying the *authority*) and Recitals 85–87, which match the topic but not the negation.
- Q4: baseline top 3 were Art. 37(4) and Art. 38 (DPO tasks), which made the DPO sound optional.
- No outright hallucination: the "answer only from context" prompt made the LLM say "not in context" instead of inventing rules, so failures show up as omissions or misstatements.

### Limitations observed

✍️ *Write briefly in your own words.*

Notes:
- Re-ranking only reorders the top 10, so recall is capped by the bi-encoder. Art. 17(3) itself (chunk 329: #7 → #6) and Art. 37(1)(a)/(b) (chunk 391: #5 → #7) never reached the top 3. Q2 was answered from the recital instead, and Q4's Answer 2 is still incomplete.
- Fixed-size chunks cut lists: Art. 34(3)(c) fell outside chunk 378, so Answer 2 for Q3 lists only (a) and (b).

---

## 3. Latency Profiling

CPU; 20 runs × 5 queries after a warm-up run. LLM time is the mean of the real Gemini calls.

| Pipeline | Retrieval | Re-rank | Retrieval total | LLM call |
|---|---|---|---|---|
| 1 – Baseline | 6.6 ms | – | 6.6 ms | ~1.56 s |
| 2 – Re-ranking | 6.6 ms | 20.2 ms | 26.8 ms | ~1.76 s |

✍️ *Interpret in 2–3 sentences.*

Notes:
- Re-ranking makes retrieval ~4× slower, but that's ~1% of the end-to-end time; the LLM call dominates.
- The LLM difference between rows comes from API/network variance, not from the pipeline.

---

## 4. Scaling Question

Measured: cross-encoding **all 580 chunks** for one query took **886 ms** (~1.5 ms per pair, ~45× the 10-candidate re-rank).

✍️ *Write your explanation in your own words.*

Notes:
- Cost is linear in corpus size and paid on **every** query. At ~1.5 ms per pair, 1M chunks ≈ 25 minutes per query.
- Bi-encoder document vectors are built once, offline; query time is one encoding plus an index lookup, and approximate indexes (HNSW/IVF) stay fast at millions of vectors.
- Two-stage design: the cheap bi-encoder handles recall (narrow to ~10), the expensive cross-encoder handles precision (order those few).
