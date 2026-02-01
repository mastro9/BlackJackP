import pickle
import random
import numpy as np
from blackjack_env import BlackjackEnv

# --- 1. OPTIMIZED CONFIGURATION ---
EPISODES = 2000000      # 2 Million are enough if the formula is correct
ALPHA = 0.001           # VERY low learning rate to stabilize values
GAMMA = 0.90            # Let's lower the importance of the future a bit
EPSILON = 1.0           
EPSILON_DECAY = 0.999995 
MIN_EPSILON = 0.05      # Always keep some exploration

env = BlackjackEnv()
q_table = {} 

print(f"Starting CORRECT training ({EPISODES} rounds)...")

for episode in range(EPISODES):
    state = env.reset()
    done = False

    while not done:
        # Initialize state if new
        if state not in q_table:
            q_table[state] = [0.0, 0.0]

        # Epsilon-Greedy
        if random.random() < EPSILON:
            action = random.choice([0, 1])
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done = env.step(action)
        
        # Initialize next state
        if next_state not in q_table:
            q_table[next_state] = [0.0, 0.0]

        # --- THE FUNDAMENTAL FIX ---
        old_value = q_table[state][action]
        next_max = np.max(q_table[next_state])
        
        # IF THE GAME IS OVER, THE FUTURE IS WORTH 0
        if done:
            target = reward
        else:
            target = reward + GAMMA * next_max

        # Updated formula
        new_value = old_value + ALPHA * (target - old_value)
        q_table[state][action] = new_value

        state = next_state

    # Decay
    if EPSILON > MIN_EPSILON:
        EPSILON *= EPSILON_DECAY

    if episode % 500000 == 0:
        print(f"Completed: {episode}, Epsilon: {EPSILON:.4f}")

print("Training completed!")

# --- FINAL VERIFICATION ---
print("\n--- BRAIN VERIFICATION (Expected values between -1 and 1) ---")
test_state = (16, 6, False) # Hard 16 vs 6

if test_state in q_table:
    values = q_table[test_state]
    stand_val = values[0]
    hit_val = values[1]
    best = "STAND" if stand_val > hit_val else "HIT"
    
    print(f"Test 16 vs 6 (Hard):")
    print(f"  Stand Value: {stand_val:.4f} (Should be slightly negative, e.g., -0.1)")
    print(f"  Hit Value: {hit_val:.4f} (Should be very negative, e.g., -0.4)")
    print(f"  -> AI chooses: {best}")
    
    if best == "STAND":
        print("CORRECT! Now it reasons well.")
    else:
        print("ERROR! Still too aggressive.")
else:
    print("State not visited.")

with open("training/blackjack_qtable.pkl", "wb") as f:
    pickle.dump(q_table, f)
print("\nSaved.")