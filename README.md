# Security-Aware Context Screening and Evidence Selection for RAG Systems
*Defending Against Indirect Prompt Injection and Knowledge Poisoning*

The initial work focuses on **dataset preparation** and **contextual evidence evaluation** using two approaches: `Filter v1` and `Filter v2`.

---

### 1. Dataset Preparation

- Used BEIR Natural Questions (NQ) for questions, Wikipedia passages, and relevance judgments.
- Used NQ Open for candidate answers and answer variants.
- Matched **2,255 questions** across the two datasets.
- Identified and validated **2,816 unique relevant passages**.
- Prepared **5,079 question–answer–passage evaluation pairs**.

---

### 2. Filter v1 — Deterministic Answer Matching

Filter v1 uses normalized answer-string containment to identify whether a candidate answer occurs in a relevant passage body.

Results:

- 2,214 questions accepted.
- 41 questions rejected.

Filter v1 provides a simple exact-containment baseline. It does not establish whether a passage genuinely supports an answer.

---

### 3. Filter v2 — Contextual Evidence Evaluation

Filter v2 uses an LLM-based contextual judge to classify the relationship between a question, candidate answer, and passage.

It supports seven evidence labels:

- `direct_support`
- `semantic_support`
- `partial_support`
- `incidental_match`
- `contradictory`
- `no_support`
- `ambiguous`

The evaluation also includes structured JSON validation, exact evidence-quote verification, retry handling, and a `review_required` outcome for responses that fail validation.

---

### 4. Human Annotation and Evaluation

A question-level split was created using seed 42:

| Split | Questions | Evaluation pairs |
|---|---:|---:|
| DEV | 50 | 97 |
| TEST | 150 | 334 |

Human annotations were used as reference labels for evaluating the contextual judge.

**Recorded Filter v2 results:**

| Evaluation | Exact agreement |
|---|---:|
| DEV — Groq | 54.95% |
| DEV — Gemini* | 53.13% |
| TEST — Gemini | **66.06%** |

*The DEV Gemini figure uses the available earlier evaluation artifact containing 96 successful predictions.*

For TEST, 327 of 334 pairs received accepted classifications. Seven responses required manual review because their evidence quotes failed exact-substring validation.

These results measure agreement with the available human labels; they do not establish perfect evidence verification or a proven security defense.

---

> [!NOTE]
> Detailed Walkthrough: [Data Preparation And Context Evaluation Report](./notes/Data_Preparation_And_Contextual_Evaluation.md).

---

## Stack

- Python
- BEIR Natural Questions
- NQ Open
- LLM providers: Groq, Gemini, and Ollama
- pytest for testing

## Project Structure

```text
config/          Configuration files
data/            Processed datasets and evaluation artifacts
scripts/         Data preparation and analysis scripts
src/             Core implementation
tests/           Automated tests
```

## Research Scope

This stage establishes the dataset and contextual evaluation foundation for the broader RAG security research project.

The next stage focuses on implementing a clean RAG retrieval baseline before introducing attacks, screening mechanisms, and defense experiments.

## Notes

- Original datasets are not included in this repository and must be obtained from their respective sources.
- API credentials must be provided through environment variables, not committed to Git.
- Human annotations serve as the reference labels for the reported evaluation.
- Evaluation results should be interpreted alongside class-wise performance, dataset limitations, and validation failures.