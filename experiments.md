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


# Passkey retrieval

## LongChat-7B-v1.5-32K 

| Method / Budget | 32 | 64 | 128 |
| :--- | :--- | :--- | :--- |
| **Quest (from the paper)** | 65% | 99% | 99% |
| **Ours (iteration 20 times)** | 0% | 0% | 100% |

##  Yarn-Llama-2-7b-128k 

sequence length 100k, OOM

| Method / Budget | 256 | 512 | 1024 |
| :--- | :--- | :--- | :--- |
| **Quest (from the paper)** | 88% | 92% | 96% |
| **Ours (iteration 50 times)** | 0% | 0% | 0% |

# LongBench

LongChat-7B-v1.5-32K
- qasper: single-document QA
- narrativeqa: single-document QA
- hotpotqa: multi-document QA
- multifieldqa_en: multi-document QA
- gov_report: text summarization
- triviaqa: few-show learning
- PassageRetrieval:(from ArkVale) not doing 