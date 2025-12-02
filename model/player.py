import pygame
import pickle  # <--- AI MOD: Import necessario
import os      # <--- AI MOD: Per trovare il file
from utils.constants import WHITE, BLUE, FONT_NORMAL, ORANGE, FONT_BOLD # Assicurati di avere un colore/font visibile
from utils.helpers import draw_text, load_image
import sys

# --- AI MOD: CARICAMENTO CERVELLO ---
# Cerchiamo di caricare la Q-Table una volta sola quando importiamo il modulo
q_table = {}
# Percorso relativo: assumiamo che la cartella 'training' sia nella root del progetto
path_to_brain = os.path.join("training", "blackjack_qtable.pkl")

try:
    with open(path_to_brain, "rb") as f:
        q_table = pickle.load(f)
    print(f"AI: Cervello caricato da {path_to_brain}")
except FileNotFoundError:
    print(f"AI: ATTENZIONE! File {path_to_brain} non trovato. L'AI non darà consigli.")
# ------------------------------------

class Player:

    def __init__(self, name):
        self.name = name

        self.hand = []
        self.count = 0

        self.blackjack = False
        self.bust = False

        self.bank = 100
        self.bet = 0

        self.x = 0
        self.y = 0
        self.currentTurn = False

        self.is_human = False

    def askChoice(self):
        """Ritorna 1 = HIT, 2 = PASS."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h: return 1
                    if event.key == pygame.K_p: return 2
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

    def addCard(self, card):
        self.hand.append(card)
        self.countCards()
    
    def countCards(self):
        self.count = 0
        for card in self.hand:
            self.count += card.value
        for card in self.hand:
            if card.label == "A":
                self.count += 10
                if self.count > 21:
                    self.count -= 10
                    break

    def applyBet(self, factor):
        self.bank += self.bet * factor

    def resetBet(self):
        self.bet = 0

    def resetHandAndCount(self):
        self.hand = []
        self.count = 0

    # --- AI MOD: Aggiunto argomento 'dealer_card' opzionale ---
    def drawHand(self, surface, dealer_card=None):
        card_w, card_h = 78, 120
        gap = 20

        start_x = self.x - (card_w + gap * (len(self.hand)-1)) / 2
        start_y = self.y - card_h / 2

        for card in self.hand:
            path = f"Resources/Cards/{card.suit}/{card.label}.png"
            img = load_image(path, (card_w, card_h))
            surface.blit(img, (start_x, start_y))
            start_x += gap

        # effetto turno
        name_color = BLUE if self.currentTurn else WHITE

        # nome + soldi
        draw_text(surface, f"{self.name}   ${self.bank}", FONT_NORMAL, name_color,
                  self.x, self.y + card_h * 0.75)

        # messaggi grafici
        if self.currentTurn and self.is_human:
            draw_text(surface, "Hit(H) or Pass(P)", FONT_NORMAL, name_color,
                      self.x, self.y - card_h * 0.75)

            # --- AI MOD: VISUALIZZAZIONE SUGGERIMENTO ---
            if dealer_card and q_table:
                suggerimento = self.get_ai_advice(dealer_card)
                # Disegna il suggerimento un po' sopra le carte (o dove preferisci)
                draw_text(surface, suggerimento, FONT_BOLD, ORANGE, self.x, self.y - card_h - 10)
            # -------------------------------------------

        if self.bust:
            bust = load_image("Resources/Icons/bust.png")
            surface.blit(bust, (self.x - bust.get_width()/2, self.y - bust.get_height()/2))

        if self.blackjack:
            bj = load_image("Resources/Icons/blackjack.png")
            surface.blit(bj, (self.x - bj.get_width()/2, self.y - bj.get_height()/2))

    def resetState(self):
        self.bust = False
        self.blackjack = False
        self.resetBet()
        self.resetHandAndCount()

    def autoChoice(self, dealer_card_value):
        # ... (il tuo codice esistente per i bot) ...
        if self.count <= 11: return 1
        if dealer_card_value >= 7 and self.count < 17: return 1
        if dealer_card_value <= 6 and self.count >= 12: return 0
        if self.count >= 17: return 0
        return 1

    # --- AI MOD: LOGICA TRADUZIONE ---
    def get_ai_advice(self, dealer_card):
        """
        Trasforma le carte attuali nella tupla (Somma, Dealer, Asso)
        e interroga la Q-Table.
        """
        # 1. Somma Giocatore
        player_sum = self.count

        # 2. Carta Dealer (Valore numerico)
        d_val = dealer_card.value
        # Nota: Nel tuo deck.py J,Q,K valgono già 10 e A vale 1. È perfetto.
        
        # 3. Asso Usabile
        # Logica: se la somma calcolata (self.count) è maggiore della somma "grezza"
        # delle carte (dove Asso vale 1), significa che stiamo usando un Asso come 11.
        raw_sum = sum(c.value for c in self.hand)
        usable_ace = (self.count > raw_sum)

        # Creiamo lo stato
        state = (player_sum, d_val, usable_ace)

        # Interroghiamo la tabella
        if state in q_table:
            valori = q_table[state]
            # valori[0] = Stand, valori[1] = Hit
            prob_stand = valori[0]
            prob_hit = valori[1]

            print("hit", prob_hit,"stand",prob_stand)
            
            migliore = "CARTA (Hit)" if prob_hit > prob_stand else "STARE (Stand)"
            
            # (Opzionale) Calcolo confidenza
            # return f"AI: {migliore} ({max(prob_hit, prob_stand):.2f})"
            return f"AI Suggerisce: {migliore}"
        else:
            return "AI: ???"
