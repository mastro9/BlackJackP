import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os

# Importiamo i tuoi file
from model import BlackjackNet
from generate_dataset import generate_dataset_fast

# --- 1. CONFIGURAZIONE ---
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DATASET_FILE = "training/blackjack_dataset.csv"
MODEL_FILE = "training/blackjack_neural_net.pth"

# --- 2. GENERAZIONE DATI (Se non esistono) ---
if not os.path.exists(DATASET_FILE):
    print("Generazione dataset in corso...")
    generate_dataset_fast(DATASET_FILE, samples=50000)
else:
    print(f"Dataset {DATASET_FILE} trovato.")

# --- 3. PREPARAZIONE DATI ---
print("Caricamento dati...")
df = pd.read_csv(DATASET_FILE)

# Input: Player Total, Dealer Card, Soft Hand (convertito a float 0.0 o 1.0)
X = df[['player_total', 'dealer_card', 'soft']].values.astype(np.float32)
# Target 1: Probabilità di vittoria (Regressione)
y_win = df['win_prob'].values.astype(np.float32).reshape(-1, 1)
# Target 2: Miglior mossa (Classificazione: 0=Stand, 1=Hit)
y_action = df['best_move'].values.astype(np.int64)

# Conversione in tensori PyTorch
tensor_x = torch.from_numpy(X)
tensor_y_win = torch.from_numpy(y_win)
tensor_y_action = torch.from_numpy(y_action)

dataset = TensorDataset(tensor_x, tensor_y_win, tensor_y_action)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 4. INIZIALIZZAZIONE MODELLO ---
model = BlackjackNet()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Loss functions: MSE per la probabilità, CrossEntropy per l'azione (Hit/Stand)
criterion_win = nn.MSELoss()
criterion_action = nn.CrossEntropyLoss()

loss_history = []

print(f"Inizio addestramento su {len(df)} esempi per {EPOCHS} epoche...")

# --- 5. LOOP DI ADDESTRAMENTO ---
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for batch_idx, (data, target_win, target_action) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        pred_win, pred_action = model(data)
        
        # Calcolo errore
        loss_w = criterion_win(pred_win, target_win)
        loss_a = criterion_action(pred_action, target_action)
        
        # Somma delle loss (vogliamo che impari entrambe le cose)
        total_loss = loss_w + loss_a
        
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()

    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)
    
    if (epoch+1) % 5 == 0:
        print(f"Epoca {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

print("Addestramento completato!")

# --- 6. SALVATAGGIO ---
torch.save(model.state_dict(), MODEL_FILE)
print(f"Modello salvato in {MODEL_FILE}")

# --- 7. GRAFICO DELL'APPRENDIMENTO ---
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Errore Totale (Loss)')
plt.title('Curva di Apprendimento Rete Neurale')
plt.xlabel('Epoche')
plt.ylabel('Errore')
plt.legend()
plt.grid(True)
plt.savefig("training_loss_graph.png")
print("Grafico salvato come training_loss_graph.png")
plt.show()