import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os

# Import your files
from model import BlackjackNet
from generate_dataset import generate_dataset_fast

# --- 1. CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DATASET_FILE = "training/blackjack_dataset.csv"
MODEL_FILE = "training/blackjack_neural_net.pth"

# --- 2. DATA GENERATION (If not existing) ---
if not os.path.exists(DATASET_FILE):
    print("Generating dataset...")
    generate_dataset_fast(DATASET_FILE, samples=50000)
else:
    print(f"Dataset {DATASET_FILE} found.")

# --- 3. DATA PREPARATION ---
print("Loading data...")
df = pd.read_csv(DATASET_FILE)

# Input: Player Total, Dealer Card, Soft Hand (converted to float 0.0 or 1.0)
X = df[['player_total', 'dealer_card', 'soft']].values.astype(np.float32)
# Target 1: Win Probability (Regression)
y_win = df['win_prob'].values.astype(np.float32).reshape(-1, 1)
# Target 2: Best Move (Classification: 0=Stand, 1=Hit)
y_action = df['best_move'].values.astype(np.int64)

# Conversion to PyTorch tensors
tensor_x = torch.from_numpy(X)
tensor_y_win = torch.from_numpy(y_win)
tensor_y_action = torch.from_numpy(y_action)

dataset = TensorDataset(tensor_x, tensor_y_win, tensor_y_action)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 4. MODEL INITIALIZATION ---
model = BlackjackNet()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Loss functions: MSE for probability, CrossEntropy for action (Hit/Stand)
criterion_win = nn.MSELoss()
criterion_action = nn.CrossEntropyLoss()

loss_history = []

print(f"Starting training on {len(df)} examples for {EPOCHS} epochs...")

# --- 5. TRAINING LOOP ---
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for batch_idx, (data, target_win, target_action) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        pred_win, pred_action = model(data)
        
        # Error calculation
        loss_w = criterion_win(pred_win, target_win)
        loss_a = criterion_action(pred_action, target_action)
        
        # Sum of losses (we want it to learn both things)
        total_loss = loss_w + loss_a
        
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()

    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

print("Training completed!")

# --- 6. SAVING ---
torch.save(model.state_dict(), MODEL_FILE)
print(f"Model saved in {MODEL_FILE}")

# --- 7. LEARNING GRAPH ---
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Total Error (Loss)')
plt.title('Neural Network Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.savefig("training_loss_graph.png")
print("Graph saved as training_loss_graph.png")
plt.show()