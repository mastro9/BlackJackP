import pickle
import random
import numpy as np
from blackjack_env import BlackjackEnv

# --- 1. CONFIGURAZIONE OTTIMIZZATA ---
EPISODES = 2000000      # 2 Milioni bastano se la formula è giusta
ALPHA = 0.001           # Learning rate MOLTO basso per stabilizzare i valori
GAMMA = 0.90            # Abbassiamo un po' l'importanza del futuro
EPSILON = 1.0           
EPSILON_DECAY = 0.999995 
MIN_EPSILON = 0.05      # Manteniamo un po' di esplorazione sempre

env = BlackjackEnv()
q_table = {} 

print(f"Inizio training CORRETTO ({EPISODES} round)...")

for episode in range(EPISODES):
    state = env.reset()
    done = False

    while not done:
        # Inizializza stato se nuovo
        if state not in q_table:
            q_table[state] = [0.0, 0.0]

        # Epsilon-Greedy
        if random.random() < EPSILON:
            action = random.choice([0, 1])
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done = env.step(action)
        
        # Inizializza prossimo stato
        if next_state not in q_table:
            q_table[next_state] = [0.0, 0.0]

        # --- IL FIX FONDAMENTALE ---
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])
        
        # SE LA PARTITA È FINITA, IL FUTURO VALE 0
        if done:
            target = reward
        else:
            target = reward + GAMMA * next_max

        # Formula aggiornata
        new_value = old_value + ALPHA * (target - old_value)
        q_table[state][action] = new_value

        state = next_state

    # Decay
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY

    if episode % 500000 == 0:
        print(f"Completati: {episode}, Epsilon: {EPSILON:.4f}")

print("Allenamento completato!")

# --- VERIFICA FINALE ---
print("\n--- VERIFICA CERVELLO (Valori attesi tra -1 e 1) ---")
test_state = (16, 6, False) # Hard 16 vs 6

if test_state in q_table:
    valori = q_table[test_state]
    stand_val = valori[0]
    hit_val = valori[1]
    migliore = "STARE" if stand_val > hit_val else "CARTA"
    
    print(f"Test 16 vs 6 (Hard):")
    print(f"  Valore Stare: {stand_val:.4f} (Dovrebbe essere leggermente negativo, es. -0.1)")
    print(f"  Valore Carta: {hit_val:.4f} (Dovrebbe essere molto negativo, es. -0.4)")
    print(f"  -> L'AI sceglie: {migliore}")
    
    if migliore == "STARE":
        print("✅ CORRETTO! Ora ragiona bene.")
    else:
        print("❌ ERRORE! Ancora aggressiva.")
else:
    print("⚠️ Stato non visitato.")

with open("training/blackjack_qtable.pkl", "wb") as f:
    pickle.dump(q_table, f)
print("\nSalvato.")