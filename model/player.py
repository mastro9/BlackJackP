import pygame
import pickle
import os
import sys
from utils.constants import WHITE, BLUE, FONT_NORMAL
from utils.helpers import draw_text, load_image

# --- CARICAMENTO CERVELLO AI ---
q_table = {}
path_to_brain = os.path.join("training", "blackjack_qtable.pkl")

try:
    with open(path_to_brain, "rb") as f:
        q_table = pickle.load(f)
    print(f"AI: Cervello caricato da {path_to_brain}")
except FileNotFoundError:
    print(f"AI: ATTENZIONE! File {path_to_brain} non trovato.")
# -------------------------------

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
    
    def resetState(self):
        self.bust = False
        self.blackjack = False
        self.resetBet()
        self.resetHandAndCount()

    def autoChoice(self, dealer_card_value):
        if self.count <= 11: return 1
        if dealer_card_value >= 7 and self.count < 17: return 1
        if dealer_card_value <= 6 and self.count >= 12: return 0
        if self.count >= 17: return 0
        return 1

    # --- NUOVA FUNZIONE: LOGICA AI ---
    def get_ai_advice(self, dealer_card):
        player_sum = self.count
        d_val = dealer_card.value
        raw_sum = sum(c.value for c in self.hand)
        usable_ace = (self.count > raw_sum)
        state = (player_sum, d_val, usable_ace)

        if state in q_table:
            valori = q_table[state]
            prob_stand = (valori[0] + 1) / 2 * 100
            prob_hit = (valori[1] + 1) / 2 * 100
            
            if prob_hit > prob_stand:
                return "HIT", prob_hit, (50, 255, 50) # Verde
            else:
                return "STAND", prob_stand, (255, 80, 80) # Rosso
        else:
            return None, 0, (200, 200, 200)

    # --- DISEGNO BADGE ---
    def draw_ai_badge(self, surface, mossa, prob, x, y, color):
        # 1. Configurazione Font (Due dimensioni diverse)
        try:
            font_title = pygame.font.SysFont("Arial", 16, bold=True) # Per la mossa
            font_sub = pygame.font.SysFont("Arial", 12)             # Per la percentuale
        except:
            font_title = pygame.font.Font(None, 20)
            font_sub = pygame.font.Font(None, 16)

        # 2. Creazione delle Scritte (Surface)
        # Riga 1: La Mossa
        text_mossa = font_title.render(f"Advice: {mossa}", True, (255, 255, 255))
        # Riga 2: La Probabilità (in grigio chiaro per contrasto)
        text_prob = font_sub.render(f"Win rate: {prob:.1f}%", True, (220, 220, 220))

        # 3. Calcolo Dimensioni del Box
        # Larghezza: prende la scritta più lunga + un po' di margine
        box_width = max(text_mossa.get_width(), text_prob.get_width()) + 20
        # Altezza: somma delle due scritte + spazi
        box_height = text_mossa.get_height() + text_prob.get_height() + 15

        # 4. Creazione Rettangolo Sfondo
        center_x = int(x)
        center_y = int(y)
        box_rect = pygame.Rect(0, 0, box_width, box_height)
        box_rect.center = (center_x, center_y)

        # 5. Disegno Sfondo Scuro
        s = pygame.Surface((box_width, box_height))
        s.set_alpha(210) # Leggermente più opaco
        s.fill((0, 0, 0))
        surface.blit(s, box_rect.topleft)

        # 6. Disegno Bordo Colorato
        try:
            pygame.draw.rect(surface, color, box_rect, width=2, border_radius=8)
        except TypeError:
            pygame.draw.rect(surface, color, box_rect, width=2)

        # 7. Posizionamento e Disegno del Testo
        # Centra la prima riga nella parte alta del box
        rect_mossa = text_mossa.get_rect(centerx=box_rect.centerx, top=box_rect.top + 5)
        surface.blit(text_mossa, rect_mossa)

        # Centra la seconda riga subito sotto la prima
        rect_prob = text_prob.get_rect(centerx=box_rect.centerx, top=rect_mossa.bottom + 2)
        surface.blit(text_prob, rect_prob)

    # --- DISEGNO MANO  ---
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

        name_color = BLUE if self.currentTurn else WHITE
        draw_text(surface, f"{self.name}   ${self.bank}", FONT_NORMAL, name_color,
                  self.x, self.y + card_h * 0.75)

        if self.currentTurn and self.is_human:
            draw_text(surface, "Hit(H) or Pass(P)", FONT_NORMAL, name_color,
                      self.x, self.y - card_h * 0.75)

            # --- CHIAMATA AI MODIFICATA ---
            if dealer_card and q_table:
                # Otteniamo i dati grezzi
                mossa, prob, colore = self.get_ai_advice(dealer_card)
                
                if mossa:
                    # Passiamo mossa e probabilità separatamente
                    # Nota: Ho alzato la posizione a -140 perché il box è più grande
                    self.draw_ai_badge(surface, mossa, prob, self.x, self.y - 140, colore)

        if self.bust:
            bust = load_image("Resources/Icons/bust.png")
            surface.blit(bust, (self.x - bust.get_width()/2, self.y - bust.get_height()/2))

        if self.blackjack:
            bj = load_image("Resources/Icons/blackjack.png")
            surface.blit(bj, (self.x - bj.get_width()/2, self.y - bj.get_height()/2))