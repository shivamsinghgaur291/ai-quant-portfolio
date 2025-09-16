# train.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from .models import MultiAssetLSTM, MultiAssetTransformer
import joblib

def create_sequences(X, y, lookback):
    xs, ys = [], []
    for i in range(len(X) - lookback):
        xs.append(X.iloc[i:i+lookback].values)   # ✅ use iloc
        ys.append(y.iloc[i+lookback].values)     # ✅ use iloc
    return np.stack(xs), np.stack(ys)


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=10, device='cpu'):
    criterion = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)
    best_val = float('inf')
    patience_ctr = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval()
        val_loss = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float(); yb = yb.to(device).float()
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
                count += 1
        val_loss /= max(count,1)
        print(f"Epoch {epoch} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping")
                break
    # load best
    model.load_state_dict(torch.load('best_model.pth'))
    return model

# Example runner function (to be called from notebook)
def fit_pipeline(X_df, y_df, lookback=60, batch_size=32, model_type='transformer', device='cpu'):
    Xseq, yseq = create_sequences(X_df, y_df, lookback)
    # train-test-val split
    n = len(Xseq)
    train_n = int(n*0.7)
    val_n = int(n*0.1)
    X_train, y_train = Xseq[:train_n], yseq[:train_n]
    X_val, y_val = Xseq[train_n:train_n+val_n], yseq[train_n:train_n+val_n]
    X_test, y_test = Xseq[train_n+val_n:], yseq[train_n+val_n:]
    # scale features using training scaler
    nsamples, seq_len, nfeat = X_train.shape
    scaler = joblib.dump({'scaler': None}, 'tmp.joblib')  # placeholder if you like; use sklearn scaler per-feature as needed
    # convert to torch
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    output_dim = y_train.shape[1]
    if model_type == 'lstm':
        model = MultiAssetLSTM(input_dim=nfeat, hidden_dim=50, num_layers=2, output_dim=output_dim, dropout=0.2)
    else:
        model = MultiAssetTransformer(input_dim=nfeat, model_dim=64, nhead=8, num_layers=2, dim_feedforward=128, output_dim=output_dim, dropout=0.1)
    model = train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=10, device=device)
    return model, (X_test, y_test)
