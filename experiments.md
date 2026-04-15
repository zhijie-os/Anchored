# Task Accuracy

- LongBench
- L-Eval
- NIAH & Passkey Retrieval

# System Efficiency

- Decoding Latency (speedup)
- KV Cache Memory Footprint

# Models and Baseline

- Yarn-Mistral-7B-128K (For extreme long-context)
- ChatGLM3-6B-128K (For varied architectures)
- Llama-3-8B-Instruct (8K context)
- Llama-2-7B-80K-Yarn


# LongChat-7B-v1.5-32K passkey retrieval

| Method / Budget | 32 | 64 | 128 |
| :--- | :--- | :--- | :--- |
| **Quest** | 65% | 99% | 99% |
| **Ours** | 0% | 0% | 100% |