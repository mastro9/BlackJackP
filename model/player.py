import pygame
import pickle
import os
import sys
from utils.constants import WHITE, BLUE, FONT_NORMAL, ORANGE, FONT_BOLD
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

    # --- NUOVA FUNZIONE MANCANTE: DISEGNO BADGE ---
    def draw_ai_badge(self, surface, text, x, y, color):
        try:
            font = pygame.font.SysFont("Arial", 16, bold=True)
        except:
            font = pygame.font.Font(None, 16)
            
        text_surf = font.render(text, True, (255, 255, 255))
        
        padding_x = 20
        padding_y = 10
        box_width = text_surf.get_width() + padding_x
        box_height = text_surf.get_height() + padding_y
        
        # IMPORTANTE: Convertiamo in int per evitare crash
        center_x = int(x)
        center_y = int(y)
        
        box_rect = pygame.Rect(0, 0, box_width, box_height)
        box_rect.center = (center_x, center_y)
        
        # Sfondo scuro trasparente
        s = pygame.Surface((box_width, box_height))
        s.set_alpha(200)
        s.fill((0, 0, 0))
        surface.blit(s, box_rect.topleft)
        
        # Bordo colorato (safe mode)
        try:
            pygame.draw.rect(surface, color, box_rect, width=2, border_radius=10)
        except TypeError:
            pygame.draw.rect(surface, color, box_rect, width=2)
        
        text_rect = text_surf.get_rect(center=box_rect.center)
        surface.blit(text_surf, text_rect)

    # --- DISEGNO MANO (Aggiornato) ---
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

            # Chiamata al badge AI
            if dealer_card and q_table:
                mossa, prob, colore = self.get_ai_advice(dealer_card)
                if mossa:
                    badge_text = f"AI: {mossa} ({prob:.1f}%)"
                    self.draw_ai_badge(surface, badge_text, self.x, self.y - 130, colore)

        if self.bust:
            bust = load_image("Resources/Icons/bust.png")
            surface.blit(bust, (self.x - bust.get_width()/2, self.y - bust.get_height()/2))

        if self.blackjack:
            bj = load_image("Resources/Icons/blackjack.png")
            surface.blit(bj, (self.x - bj.get_width()/2, self.y - bj.get_height()/2))