# Explainable Multimodal Information Diffusion Reranking via Socially-Aware Multi-Agent Reasoning



## 📖 Overview
Predicting how information spreads across social networks is a complex challenge involving multimodal content, individual user preferences, and dynamic social topologies. Traditional LLM/MLLM approaches and ID-matching models often lack interpretability and struggle with reasoning degradation when faced with massive, entangled context.

This repository contains the official implementation of **DiffAgent**, a novel **Socially-Aware Multi-Agent architecture** based on the ReAct (Reasoning and Acting) framework. By decoupling the prediction task into specialized "Expert Agents" orchestrated by a central "Coordinator," DiffAgent bridges the semantic-social gap, dynamically resolving complex diffusion pathways with high transparency and interrogatability.

## ✨ Key Features
* **Interrogatable Prediction**: Moves beyond black-box ID-matching, providing a transparent, step-by-step reasoning loop for why a user will (or will not) participate in a cascade.
* **Decoupled Expert System**: Replaces monolithic prompts with specialized agents (`Semantic`, `Profile`, and `Topology`) handling multimodal topics, user history, and graph structures independently.
* **ReAct Orchestration**: The Coordinator Agent dynamically plans, reasons, and calls specific experts to gather evidence before making a final consensus prediction.
* **Token & I/O Efficient**: Context is injected on-demand, preventing context window overflow and maximizing reasoning capabilities.

## 📂 Repository Structure

Due to GitHub's file size limitations, the `.pkl` files and multimodal resources provided here are **Test datasets** intended for testing DiffAgent. To train a model, please refer to the complete dataset (including training, validation, and test sets) at: https://www.kaggle.com/datasets/yangzhou32/omni-reldiff

```text
├── datasets/
│   ├── cascades.txt             # Full dataset: Information cascade sequences
│   ├── edges.txt                # Test dataset: Social network graph edges
│   ├── news.pkl                 # Test dataset: Multimodal topic data
│   ├── users.pkl                # Test dataset: User profiles and histories
│   ├── test.pkl                 # Test dataset: Evaluation sequences
│   └── mm/                      # Test dataset: Raw image and video files
├── baselines/
│   ├── llm_baseline.py          # Standard LLM single-prompt baseline
│   └── mllm_baseline.py         # Standard MLLM single-prompt baseline
├── DiffAgent/
│   ├── Coordinator_Agent.py     # Central ReAct orchestrator
│   ├── Semantic_Expert.py       # Parses multimodal topic motivations
│   ├── Profile_Expert.py        # Evaluates individual historical susceptibility
│   └── Topology_Expert.py       # Quantifies structural social attraction
├── utils/
│   ├── evaluation.py            # Use evaluation metrics such as HITS@k, MAP@k, and NDCG@k to evaluate the model's performance
├── main.py                      # Main evaluation pipeline for DiffAgent
├── saves/                       # Directory for output JSON results (auto-resume)
└── README.md
