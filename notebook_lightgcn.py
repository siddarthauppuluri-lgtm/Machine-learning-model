# %% [markdown]
# LightGCN product recommendation - quick notebook
# Change DATA_PATH to your CSV and run cell-by-cell.

# %%
import os
from lightgcn_recommender import DatasetLoader, train_model, evaluate_model, recommend_topk
DATA_PATH = r"F:\nit goa internship\amazon_product_review1.csv.csv"

# %%
dl = DatasetLoader(DATA_PATH)
dl.load()
train_df, val_df, test_df = dl.train_val_test_split(strategy="time_last")
print("Users:", dl.n_users, "Items:", dl.n_items)
print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))

# %%
dl.build_interaction_graph(train_df)
model = train_model(train_df, dl, epochs=3, batch_size=2048)

# %%
metrics = evaluate_model(model, dl, train_df, test_df, ks=[10,20])
print(metrics)

# %%
# sample recommendation for first user
first_user = dl.user_encoder.classes_[0]
print("Original user id:", first_user)
print("Top-10:", recommend_topk(model, dl, first_user, top_k=10))