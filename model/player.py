import pygame
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.constants import WHITE, BLUE, FONT_NORMAL, ORANGE, FONT_BOLD
from utils.helpers import draw_text, load_image

# --- 1. DEFINIZIONE DEL CERVELLO (Copiata qui per sicurezza) ---
class BlackjackNet(nn.Module):
    def __init__(self):
        super(BlackjackNet, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 32)
        self.win_head = nn.Linear(32, 1)
        self.action_head = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        win_prob = torch.sigmoid(self.win_head(x))
        action_logits = self.action_head(x)
        return win_prob, action_logits

# --- 2. CARICAMENTO MODELLO ---
MODEL_PATH = os.path.join("training", "blackjack_neural_net.pth")
ai_model = BlackjackNet()
ai_ready = False

try:
    # Carica i pesi addestrati
    ai_model.load_state_dict(torch.load(MODEL_PATH))
    ai_model.eval() # Imposta in modalità valutazione (non training)
    ai_ready = True
    print(f"AI: Rete Neurale caricata correttamente da {MODEL_PATH}")
except Exception as e:
    print(f"AI: Errore caricamento modello. Assicurati che il file .pth esista. {e}")
    ai_ready = False

# -------------------------------------------------------------

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
        # Gestione Assi
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

    # --- 3. CERVELLO PER I BOT ---
    def autoChoice(self, dealer_card_value):
        # Se l'AI non è pronta, usa una logica base di sicurezza
        if not ai_ready:
            if self.count < 17: return 1
            return 0
            
        # Usa la rete neurale per decidere
        advice, _, _ = self.get_ai_prediction(dealer_card_value)
        if advice == "HIT":
            return 1
        else:
            return 0

    # --- 4. PREVISIONE AI (Core Logic) ---
    def get_ai_prediction(self, dealer_val):
        """Prepara i dati e interroga la rete neurale"""
        if not ai_ready:
            return None, 0, (100, 100, 100)

        # 1. Calcola se la mano è "Soft" (ha un asso usabile come 11)
        raw_sum = sum(c.value for c in self.hand)
        is_soft = 1.0 if self.count > raw_sum else 0.0
        
        # 2. Prepara il tensore di input [PlayerTotal, DealerCard, Soft]
        # PyTorch vuole float32
        input_tensor = torch.tensor([[float(self.count), float(dealer_val), is_soft]])

        # 3. Chiedi al modello
        with torch.no_grad(): # Non calcolare gradiente, siamo in gioco
            win_prob, action_logits = ai_model(input_tensor)
        
        # 4. Interpreta i risultati
        # action_logits è [valore_stand, valore_hit]
        # Usiamo Softmax per trasformarli in percentuali
        probs = F.softmax(action_logits, dim=1)
        prob_stand = probs[0][0].item() * 100
        prob_hit = probs[0][1].item() * 100
        
        # Scegli l'azione migliore
        if prob_hit > prob_stand:
            return "HIT", prob_hit, (50, 255, 50) # Verde
        else:
            return "STAND", prob_stand, (255, 80, 80) # Rosso

    # --- DISEGNO BADGE (Usa la nuova funzione prediction) ---
    def get_ai_advice(self, dealer_card):
        # Questa funzione serve solo per compatibilità col vecchio codice grafico
        return self.get_ai_prediction(dealer_card.value)

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
        
        center_x = int(x)
        center_y = int(y)
        
        box_rect = pygame.Rect(0, 0, box_width, box_height)
        box_rect.center = (center_x, center_y)
        
        s = pygame.Surface((box_width, box_height))
        s.set_alpha(200)
        s.fill((0, 0, 0))
        surface.blit(s, box_rect.topleft)
        
        try:
            pygame.draw.rect(surface, color, box_rect, width=2, border_radius=10)
        except TypeError:
            pygame.draw.rect(surface, color, box_rect, width=2)
        
        text_rect = text_surf.get_rect(center=box_rect.center)
        surface.blit(text_surf, text_rect)

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

            # Badge AI (Visibile solo all'umano)
            if dealer_card:
                mossa, prob, colore = self.get_ai_advice(dealer_card)
                if mossa:
                    # Mostra "AI: HIT (98.5%)"
                    badge_text = f"NN: {mossa} ({prob:.1f}%)"
                    self.draw_ai_badge(surface, badge_text, self.x, self.y - 130, colore)

        if self.bust:
            bust = load_image("Resources/Icons/bust.png")
            surface.blit(bust, (self.x - bust.get_width()/2, self.y - bust.get_height()/2))

        if self.blackjack:
            bj = load_image("Resources/Icons/blackjack.png")
            surface.blit(bj, (self.x - bj.get_width()/2, self.y - bj.get_height()/2))