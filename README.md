# project-delta

## 🧬 Train-Tune-Deploy: Predicting Phage Lytic Functions with LLMs  
**Team Delta @ Phagos x AWS Hackdays 2025**

This repository contains the codebase, notebooks, and models developed during the **Phagos x AWS Hackdays 2025** by **Team Delta (Train Tune Deploy)**. Our goal was to identify bacteriophage proteins with strong lytic activity, leveraging **LLMs**, **transfer learning**, and **protein embeddings**, with a focus on **Klebsiella-targeting phages**.

## 🖥️ Slide Deck
 
📄 **[Click here to read our slides](https://github.com/phagos-hackathon25/project-delta/blob/main/team-delta.pdf)** presenting the full story behind our project.

#### Get a concise overview of our methodology, data insights, model architecture, saliency maps, and biological validation — all in one place.
---

### 🚀 Project Overview

#### 🧩 Problem Statement
Given a collection of phages isolated to target an unknown bacterial infection (e.g., *Klebsiella*), how can we:
- Annotate phage proteins with functional labels (lysis, replication, etc.)?
- Select candidates with strong lytic activity?
- Transfer knowledge from annotated datasets (like ESKAPE) to less-curated sets (like PHL-Klebsiella)?

#### 🎯 Our Approach
- Use **ESM-C**, a protein language model with 6B parameters, to embed amino acid sequences.
- Train a feed-forward classifier to predict 14 functional protein classes, including **Holins** and **Endolysins** (key lysis proteins).
- Visualize important residues via **saliency maps**.
- Apply the model to the **PHL-Klebsiella** dataset (Boeckaerts et al., 2024) and validate against biological assays.
- Use **entropy filtering** to identify strong candidate proteins per phage.

---

### 📂 Repository Structure

#### 🔍 Exploration & Dataset Analysis
- `explore-eskape.ipynb` — Overview of the ESKAPE dataset (14k annotated proteins).
- `explore-phl-klebsiella.ipynb` — Exploration of the Boeckaerts *Klebsiella*-phage interaction dataset.
- `explore-inphared.ipynb` — Optional exploration of the INPHARED dataset (for future extension).

#### 📊 Modeling & Visualization
- `Classifier_trained__over_Eskapet.ipynb` — Classifier training on ESKAPE embeddings.
- `t-SNE.ipynb` — 2D visualization of protein embeddings by functional class.
- `transfer_learning_over_Boeckaerts.ipynb` — Entropy-based ranking of predicted proteins for PHL-Klebsiella phages.

---

### 🧠 Key Technologies
- **ESM-C (Meta AI, 2024)** — Pretrained transformer model for protein sequence embeddings.
- **PyTorch + scikit-learn** — For model training and evaluation.
- **AWS SageMaker** — Infrastructure for scalable training and deployment (with GPU support).
- **Jupyter Notebooks** — For interactive analysis and reproducibility.

---

### 📈 Results
- Classifier trained on ESKAPE achieved **~99% accuracy** across 14 functional classes.
- Saliency maps revealed meaningful amino acid positions related to lysis.
- Transfer predictions over PHL-Klebsiella revealed **4/8 top pipeline scorers** overlapping with top biological assays.

---

### 💡 Next Steps
- Add phage-host affinity prediction.
- Extend functional labels using protein ontologies.
- Evaluate null model for strength-of-signal validation.

---

### 📚 References
- Boeckaerts et al. (2024) — *PHL-Klebsiella* Dataset.
- ESM-C (Meta AI, 2024) — 6B parameter protein language model.
- Phagos x AWS Hackdays 2025 — [Event Info](https://hackathon.phagos.org/)

---

### 🧑‍💻 Team Delta – Train Tune Deploy
- Antoine Aragon
- Emmanuel de Bézenac
- Francesco Camaglia
- Mathieu Crilout

---

### 📬 Contact
For questions or collaboration: **hackathon2025@phagos.org**
