# LLM Post-Training: SFT, GRPO, Reasoning & Continued Pretraining (vLLM + DDP)

![Status](https://img.shields.io/badge/status-research--work-blue)
![Framework](https://img.shields.io/badge/framework-Unsloth-green)
![Training](https://img.shields.io/badge/training-Continual%20Pretraining-orange)
![Language](https://img.shields.io/badge/language-Urdu-lightgrey)



A unified repository for **LLM post-training and adaptation**, covering **Supervised Fine-Tuning (SFT)**, **GRPO-based optimization**, **reasoning-focused training**, and **Continued Pretraining (CPT)** using **Unsloth**, **vLLM**, and **Distributed Data Parallel (DDP)**.

This repo is designed for researchers and engineers working on **efficient, scalable, and domain-adaptive LLM training**.

---

## 🔍 What This Repository Covers

This project focuses on **post-pretraining adaptation** of large language models, including:

- **Supervised Fine-Tuning (SFT)**
- **GRPO (Group Relative Policy Optimization)**
- **Reasoning-focused training**
- **Continued / Continual Pretraining (CPT)**
- **LoRA-based efficient finetuning**
- **Multi-GPU training with DDP**
- **Fast inference & rollout using vLLM**

Supported use cases:
- Domain adaptation (law, medicine, finance, code, etc.)
- Low-resource language adaptation
- Reasoning and alignment improvements
- Efficient finetuning of open-source LLMs

---

## 🧠 Why Continued Pretraining (CPT)?

Base models such as **LLaMA-3** or **Mistral** are pretrained on trillions of tokens, but they may still underperform on:
- Specialized domains
- New or evolving datasets
- Out-of-distribution text
- Low-resource languages

**Continued Pretraining (CPT)** allows models to keep learning from new corpora *without restarting from scratch*, making it a cost-effective and powerful adaptation strategy.

---

## 🏗️ Core Features

### ✅ Training Methods
- Supervised Fine-Tuning (SFT)
- GRPO-based optimization
- Reasoning-oriented training pipelines
- Continued Pretraining (CPT)

### ⚡ Efficiency & Scaling
- LoRA & QLoRA via **Unsloth**
- Multi-GPU training with **PyTorch DDP**
- Memory-efficient training setups
- Optimizer state reset or resume support

### 🚀 Inference & Rollouts
- Fast inference using **vLLM**
- Efficient rollout generation for GRPO
- Scalable serving-ready pipelines

---

## 🧰 Tech Stack

- **PyTorch**
- **Transformers**
- **Unsloth**
- **vLLM**
- **PEFT (LoRA / QLoRA)**
- **DDP (Distributed Data Parallel)**

---

## 📁 Repository Structure

```bash
.
├── data/
│   ├── sft/
│   ├── cpt/
│   └── reasoning/
│
├── training/
│   ├── sft/
│   ├── grpo/
│   ├── cpt/
│   └── reasoning/
│
├── inference/
│   ├── vllm/
│   └── rollout/
│
├── lora/
│   ├── adapters/
│   └── configs/
│
├── scripts/
│   ├── launch_ddp.sh
│   ├── train_sft.py
│   ├── train_grpo.py
│   └── train_cpt.py
│
├── configs/
│   ├── model.yaml
│   ├── training.yaml
│   └── grpo.yaml
│
├── requirements.txt
└── README.md
