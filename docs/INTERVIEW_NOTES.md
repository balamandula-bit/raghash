# Interview notes for RagHash

## 30-second pitch
RAG eval tools tell you an answer is wrong. RagHash tells you why: it scans a document folder for copies, old versions
and contradictions, builds a test set that follows the documents, and traces each wrong answer to the documents, the
search or the model. I built it because every "RAG chatbot" project looks the same, and the problem teams actually hit is
stale and conflicting documents.

## Design choices worth explaining
- **A fact has a key: (topic, subject, scope, region).** That one decision is why "tier-1 vs tier-2" or "India vs Germany"
  never counts as a contradiction. I built scope traps into the test data to prove it.
- **Copies must not look like new versions.** A re-saved copy has a later modified time. Version ordering collapses
  duplicate groups first and uses the original's date.
- **Similarity alone cannot separate a copy from a revision.** Near-copies and revisions overlap in text similarity, so
  the duplicate finder also requires the same numbers and the same decision words ("required", "not allowed").
  Calling a revision a duplicate is the costly mistake, so it errs the other way.
- **It says "I can't tell".** Versions with no usable date go to an `undetermined` list; dates from file modified
  times get lower confidence.
- **Blame is the product.** Each wrong answer gets a cause and an owner (documents, search, model), and a ranked list of
  files to fix.

## How I know it works
- Synthetic corpus with a hidden answer key; strict scoring where any pair outside the key is a false alarm.
- Failure injection: a fake RAG that breaks answers in 9 known ways to test the diagnoser (8,412 cases, 0 mismatches).
- Drift simulation: change 5 documents, check that exactly the 5 affected questions are flagged.
- Bugs it found in my own work: a flipped "allowed" scored as a duplicate; a doc listing two scopes fooled whole-document
  labelling (fixed by judging individual claims); a yes/no ambiguity I documented rather than hid.

## What I would say is missing (before they ask)
- The data is synthetic and templated; perfect scores prove the logic, not real-world accuracy.
- Fact extraction is pattern-based; real documents need an LLM extractor, and I would measure its error rate against a
  hand-labelled sample of real policies.
- Only .md/.txt; PDF and DOCX parsing, and chunk-level retrieval analysis, are next.
- The baseline RAG is deliberately simple; I would plug in a real LangChain or LlamaIndex pipeline through the same
  answers file.

## Likely questions
- *Why not just use Ragas?* Ragas scores answers; it does not look at the corpus. This sits on top of it.
- *Why MinHash?* Comparing every pair of documents does not scale; LSH narrowed 23,005 possible pairs to about 600.
- *Why no dependencies?* Easy to run, easy to read, and it shows the algorithms are understood, not imported.
- *What would you do with a real corpus?* Start with 50 public policy documents, hand-label 30 facts, measure the
  extractor, then run the pipeline.
