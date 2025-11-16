# Aurora
Loom Recording: https://www.loom.com/share/9adb045e05f44a88a3b8d5faf91bc03c

📌 Bonus 1 — Design Notes (Alternative Approaches)

Before selecting the final architecture, I evaluated several alternatives:

Approach	Pros	Cons	Reason not chosen
Direct LLM QA (send full data + question to GPT/Claude)	High accuracy	Data leaves system, cost, latency, hallucination risk	Would violate grounding + reliability
Keyword search + regex extraction	Simple	Fails on rephrasing and synonyms	Low accuracy
Vector DB (FAISS / Pinecone)	Real-world scalable	Infra complexity for only 100 messages	Overkill
Sentence-Transformer embeddings + custom extraction (final choice)	Fast, robust, private, deterministic	Requires extraction logic	Best balance of performance & reliability
