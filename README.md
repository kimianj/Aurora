# Aurora
Loom Recording: https://www.loom.com/share/9adb045e05f44a88a3b8d5faf91bc03c


▶️ For running the project
pip install -r requirements.txt '\n'
uvicorn main:app --host 0.0.0.0 --port 8000



API Endpoints

| Method | Endpoint            | Body / Params           | Description                             |
| ------ | ------------------- | ----------------------- | --------------------------------------- |
| `POST` | `/ask`              | `{ "question": "..." }` | Ask a question (recommended)            |
| `GET`  | `/ask?question=...` | Query param             | Convenience endpoint for manual testing |
| `GET`  | `/health`           | —                       | API health and embedding status         |
| `GET`  | `/insights`         | —                       | Dataset insights & anomaly report       |


📌 Bonus 1: Design Notes (Alternative Approaches)
Before selecting the final architecture, I evaluated several alternatives:

| Approach                                  | Pros                                    | Cons                                  |
| ----------------------------------------- | --------------------------------------- | ------------------------------------- |
| **RAG with sentence embeddings (chosen)** | Fast, cheap, deterministic, no LLM cost | Rule-based answer extraction          |
| **LLM-based QA (GPT, Claude, etc.)**      | Very natural reasoning                  | Requires API key, costs $, slower     |
| **Fine-tuned QA transformer**             | High precision                          | Requires labeled data + training time |
| **Keyword extraction + exact match**      | Very simple                             | Poor recall & reasoning               |


🔍 Bonus 2: Data Insights (Automated)

{
  "total_messages": 218,
  "unique_users": 42,
  "date_range": {
    "earliest": "2022-11-04T09:14:00Z",
    "latest": "2024-07-27T16:50:00Z"
  },
  "anomalies": [
    "3 messages missing user_name",
    "1 message missing timestamp",
    "2 messages with potential encoding issues in user names",
    "1 future timestamp detected"
  ]
}



These were observed in the dataset:
Some messages missing user_name or timestamp
A few timestamps in the future
Possible duplicate users with slight spelling variations



Author: Kimia Naeiji

