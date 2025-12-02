import pickle
import random
import numpy as np
from blackjack_env import BlackjackEnv

# --- 1. CONFIGURAZIONE POTENZIATA ---
EPISODES = 5000000      # 5 Milioni di partite (più esperienza)
ALPHA = 0.01            # Learning Rate più basso (impara con più cautela, più preciso)
GAMMA = 0.95            
EPSILON = 1.0           
EPSILON_DECAY = 0.999999 # Decadimento lentissimo per esplorare bene
MIN_EPSILON = 0.01      

env = BlackjackEnv()
q_table = {} 

print(f"Inizio allenamento intensivo ({EPISODES} round)... Attendere prego.")

# --- 2. TRAINING LOOP ---
for episode in range(EPISODES):
    state = env.reset()
    done = False

    while not done:
        # Gestione stati non ancora visitati
        if state not in q_table:
            q_table[state] = [0.0, 0.0]

        # Epsilon-Greedy Strategy
        if random.random() < EPSILON:
            action = random.choice([0, 1])
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done = env.step(action)
        
        # Inizializza prossimo stato se nuovo
        if next_state not in q_table:
            q_table[next_state] = [0.0, 0.0]

        # Q-Learning Update Rule
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])
        
        new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
        q_table[state][action] = new_value

        state = next_state

    # Aggiornamento Epsilon
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY

    # Stampa progresso ogni milione
    if episode % 1000000 == 0:
        print(f"Completati: {episode}, Epsilon: {EPSILON:.4f}")

print("Allenamento completato!")

# --- 3. VERIFICA DELLA LOGICA (SANITY CHECK) ---
print("\n--- VERIFICA CERVELLO ---")

# Caso Critico: Player 16 (Hard), Dealer 6
# Stato: (16, 6, False) -> False significa Asso non usabile (Hard)
test_state = (16, 6, False)

if test_state in q_table:
    valori = q_table[test_state]
    stand_val = valori[0]
    hit_val = valori[1]
    migliore = "STARE" if stand_val > hit_val else "CARTA"
    
    print(f"Test 16 vs 6 (Hard):")
    print(f"  Valore Stare: {stand_val:.4f}")
    print(f"  Valore Carta: {hit_val:.4f}")
    print(f"  -> L'AI sceglie: {migliore}")
    
    if migliore == "STARE":
        print("✅ CORRETTO! L'AI ha imparato.")
    else:
        print("❌ ERRORE! L'AI vuole ancora hittare. Riprova il training.")
else:
    print("⚠️ Stato non visitato abbastanza volte.")

# --- 4. SALVATAGGIO ---
with open("training/blackjack_qtable.pkl", "wb") as f:
    pickle.dump(q_table, f)
print("\nCervello salvato in 'blackjack_qtable.pkl'")