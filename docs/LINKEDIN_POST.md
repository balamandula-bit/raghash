# LinkedIn post draft

Most RAG evaluation tells you an answer is wrong. It doesn't tell you why.

So I built RagHash to answer the question teams actually ask next: is it our documents, our search, or our model?

In a company, one policy lives in a wiki page, a PDF and an old copy on a shared drive. Someone updates one. The RAG system quotes the stale one, or the FAQ that disagrees with the policy. The team then spends days tuning prompts for a problem that was in the documents all along.

What it does:
- finds copies, superseded versions and contradictions in a document folder
- builds a test set that follows the documents as they change (5 documents changed, exactly 5 of 609 questions flagged)
- traces every wrong answer to a cause and an owner: documents, search, or model
- writes one report: which files to fix first, ranked by how many questions they repair

On a plain BM25 baseline, 24% of the wrong answers traced back to the documents, and the biggest single cause was answering for the wrong region, an answering-side bug that no document fix would catch.

How I checked it: the practice corpus is synthetic with a hidden answer key, so every claim is scored. The diagnoser was tested by breaking answers on purpose in 9 ways: 8,412 injected failures, 0 wrong diagnoses. The honest part: the data is synthetic and templated, so those scores prove the logic, not real-world performance. A real deployment needs an LLM to read the facts out of messy files, and that is the next step.

Pure Python, no dependencies, 104 tests. One command runs the whole demo.

GitHub: [link]

#RAG #LLM #AIEngineering #Python #MachineLearning
