import random
import csv
from tqdm import tqdm

# ----------------------------------------
# Valore mano + rilevamento soft
# ----------------------------------------
def hand_value(cards):
    total = sum(cards)
    aces = cards.count(11)

    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    soft = (aces > 0)
    return total, soft

# ----------------------------------------
# Pesca carta (mazzo statistico infinito)
# ----------------------------------------
def draw_card():
    v = random.randint(1, 13)
    return 11 if v == 1 else 10 if v >= 10 else v

# ----------------------------------------
# Dealer veloce (early exit)
# ----------------------------------------
def play_dealer_fast(upcard):
    total = upcard + draw_card()

    # gestiamo gli assi
    if upcard == 11 or total - upcard == 11:
        soft = True
    else:
        soft = False

    # ottimizziamo: dealer pesca finché < 17
    while True:
        # Se total >= 17: fermati
        if total > 17 or (total == 17 and not soft):
            break

        card = draw_card()
        total += card

        if card == 11:
            soft = True
        if total > 21 and soft:
            total -= 10
            soft = False

        if total > 21:
            break

    return total

# ----------------------------------------
# Simula HIT e STAND in un'unica passata
# ----------------------------------------
def estimate_win_prob_fast(player_total, player_soft, dealer_up, N=1000):
    hit_wins = 0
    stand_wins = 0

    for _ in range(N):
        # ---------- STAND ----------
        dealer_total_s = play_dealer_fast(dealer_up)
        if dealer_total_s > 21 or player_total > dealer_total_s:
            stand_wins += 1

        # ---------- HIT ----------
        # Pesca una carta
        card = draw_card()
        t = player_total + card
        soft = player_soft

        # aggiorna soft se Asso
        if card == 11:
            soft = True

        # Asso aggiustato
        if t > 21 and soft:
            t -= 10
            soft = False

        # Bust = perdita immediata
        if t <= 21:
            dealer_total_h = play_dealer_fast(dealer_up)
            if dealer_total_h > 21 or t > dealer_total_h:
                hit_wins += 1

    p_hit = hit_wins / N
    p_stand = stand_wins / N

    return p_hit, p_stand

# ----------------------------------------
# Generatore dataset VELocissimo
# ----------------------------------------
def generate_dataset_fast(filename="training/blackjack_dataset.csv", samples=50000):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["player_total", "dealer_card", "soft", "win_prob", "best_move"])

        for _ in tqdm(range(samples)):
            # mano random veloce
            c1, c2 = draw_card(), draw_card()
            total = c1 + c2
            soft = (c1 == 11 or c2 == 11)

            # aggiusta se sballa
            if total > 21 and soft:
                total -= 10
                soft = False

            dealer_up = draw_card()

            # stima veloce Monte Carlo
            p_hit, p_stand = estimate_win_prob_fast(total, soft, dealer_up)

            if p_hit > p_stand:
                best_move = 1
                win_prob = p_hit
            else:
                best_move = 0
                win_prob = p_stand

            # salva riga
            writer.writerow([total, dealer_up, int(soft), win_prob, best_move])

    print(f"Dataset generato: {filename}")


if __name__ == "__main__":
    generate_dataset_fast(samples=50000)
