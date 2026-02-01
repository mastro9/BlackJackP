import random
import csv
from tqdm import tqdm

# ----------------------------------------
# Hand value + soft detection
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
# Draw card (infinite statistical deck)
# ----------------------------------------
def draw_card():
    v = random.randint(1, 13)
    return 11 if v == 1 else 10 if v >= 10 else v

# ----------------------------------------
# Fast Dealer (early exit)
# ----------------------------------------
def play_dealer_fast(upcard):
    total = upcard + draw_card()

    # handle aces
    if upcard == 11 or total - upcard == 11:
        soft = True
    else:
        soft = False

    # optimization: dealer hits until < 17
    while True:
        # If total >= 17: stop
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
# Simulate HIT and STAND in a single pass
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
        # Draw a card
        card = draw_card()
        t = player_total + card
        soft = player_soft

        # update soft if Ace
        if card == 11:
            soft = True

        # Adjusted Ace
        if t > 21 and soft:
            t -= 10
            soft = False

        # Bust = immediate loss
        if t <= 21:
            dealer_total_h = play_dealer_fast(dealer_up)
            if dealer_total_h > 21 or t > dealer_total_h:
                hit_wins += 1

    p_hit = hit_wins / N
    p_stand = stand_wins / N

    return p_hit, p_stand

# ----------------------------------------
# Very FAST Dataset Generator
# ----------------------------------------
def generate_dataset_fast(filename="training/blackjack_dataset.csv", samples=50000):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["player_total", "dealer_card", "soft", "win_prob", "best_move"])

        for _ in tqdm(range(samples)):
            # fast random hand
            c1, c2 = draw_card(), draw_card()
            total = c1 + c2
            soft = (c1 == 11 or c2 == 11)

            # adjust if bust
            if total > 21 and soft:
                total -= 10
                soft = False

            dealer_up = draw_card()

            # fast Monte Carlo estimation
            p_hit, p_stand = estimate_win_prob_fast(total, soft, dealer_up)

            if p_hit > p_stand:
                best_move = 1
                win_prob = p_hit
            else:
                best_move = 0
                win_prob = p_stand

            # save row
            writer.writerow([total, dealer_up, int(soft), win_prob, best_move])

    print(f"Dataset generated: {filename}")


if __name__ == "__main__":
    generate_dataset_fast(samples=50000)