#!/usr/bin/env python3
"""
LightGCN recommender with:
- flexible DatasetLoader that auto-detects columns
- train/val/test split (time-aware if purchase_date exists; else leave-one-out per user)
- training loop using BPR loss
- evaluation: Recall@K and NDCG@K
- FastAPI endpoint to request recommendations by original user_id or user_idx
"""
import os
import random
from typing import List, Tuple, Dict, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.nn import LGConv
from fastapi import FastAPI, HTTPException
import uvicorn
import nest_asyncio
import threading

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Dataset loader & splits
class DatasetLoader:
    def __init__(self, path: str,
                 user_col: str = "user_id",
                 item_col: str = "product_id",
                 rating_col: str = "rating",
                 returned_col: str = "is_returned",
                 time_col: str = "purchase_date"):
        self.path = path
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.returned_col = returned_col
        self.time_col = time_col
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.df = None

    def load(self) -> "DatasetLoader":
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"CSV not found at: {self.path}")
        df = pd.read_csv(self.path)
        if self.user_col not in df.columns or self.item_col not in df.columns:
            raise KeyError(f"Expected columns '{self.user_col}' and '{self.item_col}' in CSV")
        if self.rating_col in df.columns:
            try:
                df[self.rating_col] = pd.to_numeric(df[self.rating_col], errors='coerce')
                df = df[df[self.rating_col] >= 4]  
            except Exception:
                pass

        if self.returned_col in df.columns:
            df = df[df[self.returned_col] == False]
        if self.time_col in df.columns:
            try:
                df[self.time_col] = pd.to_datetime(df[self.time_col], errors='coerce')
            except Exception:
                pass
        df['_user_orig'] = df[self.user_col].astype(str)
        df['_item_orig'] = df[self.item_col].astype(str)
        df['user_idx'] = self.user_encoder.fit_transform(df['_user_orig'])
        df['item_idx'] = self.item_encoder.fit_transform(df['_item_orig'])

        self.df = df.reset_index(drop=True)
        self.n_users = int(self.df['user_idx'].nunique())
        self.n_items = int(self.df['item_idx'].nunique())
        return self

    def build_interaction_graph(self, interaction_df: pd.DataFrame) -> Data:
        """
        Build a bipartite edge_index from interactions (expects user_idx and item_idx fields).
        Items are offset by n_users.
        """
        users = interaction_df['user_idx'].values
        items = interaction_df['item_idx'].values + self.n_users
        edge_index_array = np.stack([
            np.concatenate([users, items]),
            np.concatenate([items, users])
        ])
        edge_index = torch.from_numpy(edge_index_array).long()
        data = Data(edge_index=edge_index)
        data.num_nodes = self.n_users + self.n_items
        self.data = data
        return data

    def train_val_test_split(self, strategy: str = "time_last", val_frac: float = 0.01, seed: int = SEED
                             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splitting strategies:
         - "time_last": if time_col exists, sort per user by time; last -> test, second-last -> val, rest train
         - "leave_one_out": per-user last interaction -> test, one random -> val if possible
         - "random": global random split with given val_frac (small)
        Returns: train_df, val_df, test_df
        """
        df = self.df.copy()
        if strategy == "time_last" and self.time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[self.time_col]):
            df = df.sort_values([ 'user_idx', self.time_col ])
            test_rows = []
            val_rows = []
            train_rows = []
            for uid, group in df.groupby('user_idx'):
                if len(group) == 1:
                    train_rows.append(group.index.values[0])
                elif len(group) == 2:
                    train_rows.extend(group.index.values[:-1].tolist())
                    test_rows.append(group.index.values[-1])
                else:
                    train_rows.extend(group.index.values[:-2].tolist())
                    val_rows.append(group.index.values[-2])
                    test_rows.append(group.index.values[-1])
            train_df = df.loc[train_rows].reset_index(drop=True)
            val_df = df.loc[val_rows].reset_index(drop=True) if val_rows else pd.DataFrame(columns=df.columns)
            test_df = df.loc[test_rows].reset_index(drop=True)
            return train_df, val_df, test_df

        if strategy == "leave_one_out":
            np.random.seed(seed)
            train_idx, val_idx, test_idx = [], [], []
            for uid, group in df.groupby('user_idx'):
                idxs = group.index.values
                if len(idxs) == 1:
                    train_idx.append(idxs[0])
                elif len(idxs) == 2:
                    train_idx.append(idxs[0])
                    test_idx.append(idxs[1])
                else:
                    test_i = np.random.choice(idxs, 1)[0]
                    rem = np.setdiff1d(idxs, [test_i])
                    val_i = np.random.choice(rem, 1)[0]
                    train_i = np.setdiff1d(rem, [val_i])
                    train_idx.extend(train_i.tolist())
                    val_idx.append(val_i)
                    test_idx.append(test_i)
            return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(df)
        n_val = max(1, int(n * val_frac))
        n_test = n_val
        train_df = df.iloc[: n - n_val - n_test].reset_index(drop=True)
        val_df = df.iloc[n - n_val - n_test: n - n_test].reset_index(drop=True)
        test_df = df.iloc[n - n_test:].reset_index(drop=True)
        return train_df, val_df, test_df

# Model

class AdvancedLightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, embed_size: int = 64, n_layers: int = 3, alpha: float = 0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.alpha = alpha
        self.user_emb = nn.Embedding(n_users, embed_size)
        self.item_emb = nn.Embedding(n_items, embed_size)
        self.convs = nn.ModuleList([LGConv() for _ in range(n_layers)])
        self.layer_weights = nn.Parameter(torch.ones(n_layers + 1))
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, edge_index: torch.LongTensor):
        x0 = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        x = x0
        embeddings = [x0]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x + self.alpha * x0
            embeddings.append(x)
        weights = torch.softmax(self.layer_weights, dim=0)
        final_emb = sum(w * e for w, e in zip(weights, embeddings))
        return final_emb[:self.n_users], final_emb[self.n_users:]

# Loss & evaluation helpers

def bpr_loss(user_emb, item_emb, users, pos, neg, eps: float = 1e-8):
    u = user_emb[users]
    p = item_emb[pos]
    n = item_emb[neg]
    pos_scores = (u * p).sum(dim=1)
    neg_scores = (u * n).sum(dim=1)
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + eps).mean()


def dcg_at_k(r: List[int], k: int):
    r = np.asarray(r, dtype=float)[:k]
    if r.size == 0:
        return 0.0
    return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))


def ndcg_at_k(preds: List[int], ground_truth: List[int], k: int):
    r = [1 if p in ground_truth else 0 for p in preds[:k]]
    dcg = dcg_at_k(r, k)
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    return float(dcg / idcg) if idcg > 0 else 0.0

# Training

def train_model(train_df: pd.DataFrame, dataset_loader: DatasetLoader, epochs: int = 5, batch_size: int = 4096, lr: float = 1e-3, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = AdvancedLightGCN(dataset_loader.n_users, dataset_loader.n_items, embed_size=64, n_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    
    dataset_loader.build_interaction_graph(train_df)
    edge_index = dataset_loader.data.edge_index.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        user_pos = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        
        idxs = np.arange(len(train_df))
        np.random.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            batch_idx = idxs[i:i + batch_size]
            batch = train_df.iloc[batch_idx]
            users = torch.LongTensor(batch['user_idx'].values).to(device)
            pos = torch.LongTensor(batch['item_idx'].values).to(device)
            
            neg = []
            for u, p in zip(batch['user_idx'].values, batch['item_idx'].values):
                neg_item = np.random.randint(0, dataset_loader.n_items)
                while neg_item in user_pos.get(u, set()):
                    neg_item = np.random.randint(0, dataset_loader.n_items)
                neg.append(neg_item)
            neg = torch.LongTensor(neg).to(device)

            optimizer.zero_grad()
            user_emb, item_emb = model(edge_index)
            loss = bpr_loss(user_emb, item_emb, users, pos, neg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} finished, total loss: {total_loss:.4f}")

    return model



# Evaluation

def evaluate_model(model: AdvancedLightGCN, dataset_loader: DatasetLoader, train_df: pd.DataFrame, test_df: pd.DataFrame, ks: List[int] = [10, 20], device=None) -> Dict[str, float]:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    edge_index = dataset_loader.data.edge_index.to(device)  
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(edge_index)
        item_emb = item_emb.to(device)


        train_pos = train_df.groupby('user_idx')['item_idx'].apply(set).to_dict()
        # if val exists, also mask val later if provided
        metrics = {f"Recall@{k}": 0.0 for k in ks}
        metrics.update({f"NDCG@{k}": 0.0 for k in ks})
        n_users = 0

         
        for uid, group in test_df.groupby('user_idx'):
            gt_items = group['item_idx'].unique().tolist()
            if len(gt_items) == 0:
                continue
            n_users += 1
            u_vec = user_emb[uid].unsqueeze(0)  # (1,d)
            scores = torch.matmul(u_vec, item_emb.t()).squeeze(0).cpu().numpy()  # (n_items,)
            
            seen = list(train_pos.get(uid, set()))
            if len(seen) > 0:
                scores[seen] = -1e9
            
            max_k = max(ks)
            topk = np.argpartition(-scores, max_k - 1)[:max_k]
            topk_sorted = topk[np.argsort(-scores[topk])]

            for k in ks:
                preds_k = topk_sorted[:k].tolist()

                recall = 1.0 if any([g in preds_k for g in gt_items]) else 0.0
                ndcg = ndcg_at_k(preds_k, gt_items, k)
                metrics[f"Recall@{k}"] += recall
                metrics[f"NDCG@{k}"] += ndcg

        # average
        if n_users == 0:
            raise ValueError("No users in test set for evaluation")
        for key in list(metrics.keys()):
            metrics[key] = metrics[key] / n_users
        metrics['n_eval_users'] = n_users
        return metrics



# Inference & API

def recommend_topk(model: AdvancedLightGCN, dataset_loader: DatasetLoader, user_identifier: Union[int, str], top_k: int = 10, device=None) -> List[str]:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    edge_index = dataset_loader.data.edge_index.to(device)
    
    if isinstance(user_identifier, str):
        if user_identifier not in dataset_loader.user_encoder.classes_:
            raise KeyError("User id not found")
        user_idx = int(dataset_loader.user_encoder.transform([user_identifier])[0])
    else:
        user_idx = int(user_identifier)
        if user_idx < 0 or user_idx >= dataset_loader.n_users:
            raise IndexError("user_idx out of range")
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(edge_index)
        scores = torch.matmul(user_emb[user_idx].unsqueeze(0), item_emb.t()).squeeze(0).cpu().numpy()
        
        seen_train = dataset_loader.df[dataset_loader.df['user_idx'] == user_idx]['item_idx'].unique().tolist()
        if len(seen_train) > 0:
            scores[seen_train] = -1e9
        topk_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
        return dataset_loader.item_encoder.inverse_transform(topk_idx).tolist()


app = FastAPI()
GLOBAL_MODEL = None
GLOBAL_DATASET_LOADER = None
GLOBAL_TRAIN_DF = None

@app.get("/recommend/{user_id}")
def api_recommend(user_id: str, k: int = 10):
    if GLOBAL_MODEL is None or GLOBAL_DATASET_LOADER is None:
        raise HTTPException(status_code=503, detail="Model/dataset not loaded")
    try:
        
        if user_id.isdigit() and int(user_id) < GLOBAL_DATASET_LOADER.n_users:
            uid = int(user_id)
        else:
            uid = user_id
        recs = recommend_topk(GLOBAL_MODEL, GLOBAL_DATASET_LOADER, uid, top_k=k)
        return {"user": user_id, "recommendations": recs}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





def main():
    DATA_PATH = r"F:\nit goa internship\amazon_product_review1.csv.csv"
    dl = DatasetLoader(DATA_PATH)
    dl.load()
    train_df, val_df, test_df = dl.train_val_test_split(strategy="time_last")
    
    dl.build_interaction_graph(train_df)
    model = train_model(train_df, dl, epochs=5, batch_size=2048)
    print("Evaluating on test set...")
    metrics = evaluate_model(model, dl, train_df, test_df, ks=[10, 20])
    print(metrics)

    
    global GLOBAL_MODEL, GLOBAL_DATASET_LOADER, GLOBAL_TRAIN_DF
    GLOBAL_MODEL = model
    GLOBAL_DATASET_LOADER = dl
    GLOBAL_TRAIN_DF = train_df
    nest_asyncio.apply()
    def run_uvicorn():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    t = threading.Thread(target=run_uvicorn, daemon=True)
    t.start()
    print("API running on port 8000")

if __name__ == "__main__":
    main()