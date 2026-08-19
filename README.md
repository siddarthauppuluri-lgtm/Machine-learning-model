# Product Recommendation System using LightGCN

A personalized recommendation engine built on Amazon Product Reviews using **LightGCN** (Graph Convolutional Networks). The project models user-item interactions as a bipartite graph, using neighbor aggregation and layer combination to generate top-K product recommendations.

---

## 📌 Project Overview

Standard collaborative filtering models struggle to capture higher-order relational signals across sparse user-item graphs. This system utilizes LightGCN—a streamlined Graph Neural Network tailored for recommendation tasks by removing complex non-linear feature transformations and activations.

### Key Features
* **Bipartite Interaction Graphs:** Built directly from preprocessed Amazon review datasets.
* **Over-smoothing Mitigation:** Implements residual aggregation and layer weighting across propagation steps.
* **Real-time Event Tracking:** Supports live user interactions including clicks, ratings, and purchases.
* **Performance Benchmarking:** Evaluated on standard recommendation metrics: **Precision@K**, **Recall@K**, and **NDCG@K**.

---

## 📂 Repository Structure

```text
├── amazon_product_review1.csv     # Preprocessed dataset sample
├── lightgcn_recommender.py        # LightGCN model definition and training logic
├── notebook_lightgcn.py           # Experimentation, training, and evaluation pipeline
├── file_reduction.py              # Dataset downsampling and preprocessing script
├── requirements.txt               # Project dependencies
├── Dockerfile.txt                 # Docker container configuration
└── README.md                      # Documentation


⚙️ Installation & Setup
Clone the repository:
git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git
cd <YOUR_REPOSITORY_NAME>


Create and activate a virtual environment:
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

How to Run
(Optional) Reduce/Sample Dataset:
python file_reduction.py

Train the LightGCN Recommender:

Bash
python lightgcn_recommender.py

Run Full Evaluation Pipeline:
python notebook_lightgcn.py

📊 Evaluation MetricsThe system measures ranking accuracy across top-K recommendations ($K \in \{5, 10, 20\}$):
Precision@K: Ratio of recommended items that are relevant to the user.
Recall@K: Coverage of total relevant items retrieved in top-K.
NDCG@K: Ranking quality metric penalizing relevant items appearing lower down the list.

🛠️ Tech Stack
Language: PythonData
Processing: Pandas, NumPy
Deep Learning & Graphs: PyTorch, PyTorch Geometric / DGL, Scikit-learn
Containerization: Docker


