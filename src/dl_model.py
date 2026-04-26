import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import sqlite3
import os
import pickle
import json

DB_PATH = "data/ecommerce.db"
MODEL_DIR = "models"
SEQ_LENGTH = 5

class CategoryDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx): return self.sequences[idx], self.targets[idx]

class NextCategoryMLP(nn.Module):
    def __init__(self, num_categories, embedding_dim=32, hidden_dim=64):
        super(NextCategoryMLP, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(SEQ_LENGTH * embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_categories)
    def forward(self, x):
        embedded = self.embedding(x)
        flattened = embedded.view(x.size(0), -1)
        hidden = self.dropout(self.relu(self.fc1(flattened)))
        return self.fc2(hidden)

def get_sequence_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT c.customer_unique_id, p.product_category_name, f.order_purchase_timestamp
    FROM Fact_Order_Items f
    JOIN Dim_Customers c ON f.customer_id = c.customer_id
    JOIN Dim_Products p ON f.product_id = p.product_id
    WHERE p.product_category_name != 'unknown' AND f.order_purchase_timestamp IS NOT NULL
    ORDER BY c.customer_unique_id, f.order_purchase_timestamp
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    categories = df['product_category_name'].unique().tolist()
    cat2idx = {cat: i+1 for i, cat in enumerate(categories)}
    cat2idx['<PAD>'] = 0
    idx2cat = {i: cat for cat, i in cat2idx.items()}
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "category_vocab.pkl"), 'wb') as f:
        pickle.dump((cat2idx, idx2cat), f)
        
    df['cat_idx'] = df['product_category_name'].map(cat2idx)
    sequences, targets = [], []
    for _, group in df.groupby('customer_unique_id'):
        user_seq = group['cat_idx'].tolist()
        if len(user_seq) < 2: continue
        if len(user_seq) < SEQ_LENGTH + 1:
            padded = [0] * (SEQ_LENGTH + 1 - len(user_seq)) + user_seq
            sequences.append(padded[:-1])
            targets.append(padded[-1])
        else:
            for i in range(len(user_seq) - SEQ_LENGTH):
                sequences.append(user_seq[i:i+SEQ_LENGTH])
                targets.append(user_seq[i+SEQ_LENGTH])
    return np.array(sequences), np.array(targets), len(cat2idx)

def train_dl_model():
    sequences, targets, num_categories = get_sequence_data()
    dataset = CategoryDataset(sequences, targets)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    model = NextCategoryMLP(num_categories=num_categories)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    epochs = 10
    history = {'train_loss': [], 'val_loss': []}
    
    print("Training PyTorch MLP for Next-Category prediction...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_seq, batch_tgt in train_loader:
            optimizer.zero_grad()
            out = model(batch_seq)
            loss = criterion(out, batch_tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_seq, batch_tgt in val_loader:
                out = model(batch_seq)
                loss = criterion(out, batch_tgt)
                val_loss += loss.item()
                
        t_loss_avg = train_loss/len(train_loader)
        v_loss_avg = val_loss/len(val_loader)
        history['train_loss'].append(t_loss_avg)
        history['val_loss'].append(v_loss_avg)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {t_loss_avg:.4f}, Val Loss: {v_loss_avg:.4f}")
        
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "next_cat_mlp.pth"))
    with open(os.path.join(MODEL_DIR, "dl_history.json"), 'w') as f:
        json.dump(history, f)
    print("PyTorch model and loss history saved successfully.")

def predict_next_category(cat_history_names):
    with open(os.path.join(MODEL_DIR, "category_vocab.pkl"), 'rb') as f:
        cat2idx, idx2cat = pickle.load(f)
    model = NextCategoryMLP(num_categories=len(cat2idx))
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "next_cat_mlp.pth")))
    model.eval()
    
    indices = [cat2idx.get(name, 0) for name in cat_history_names]
    indices = [0] * (SEQ_LENGTH - len(indices)) + indices if len(indices) < SEQ_LENGTH else indices[-SEQ_LENGTH:]
    seq_tensor = torch.tensor([indices], dtype=torch.long)
    with torch.no_grad():
        probs = torch.softmax(model(seq_tensor), dim=1)
        top_probs, top_idx = torch.topk(probs, 3, dim=1)
        
    results = []
    for i in range(3):
        idx = top_idx[0][i].item()
        if idx in idx2cat:
            results.append({'category': idx2cat[idx], 'probability': top_probs[0][i].item()})
    return results

if __name__ == "__main__":
    train_dl_model()
