import pygame
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.constants import WHITE, GREEN, FONT_NORMAL, ORANGE, FONT_BOLD
from utils.helpers import draw_text, load_image

# --- 1. BRAIN DEFINITION ---
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

# --- 2. MODEL LOADING ---
MODEL_PATH = os.path.join("training", "blackjack_neural_net.pth")
ai_model = BlackjackNet()
ai_ready = False

try:
    # Loads the trained weights
    ai_model.load_state_dict(torch.load(MODEL_PATH))
    ai_model.eval() # Sets evaluation mode (not training)
    ai_ready = True
    print(f"AI: Neural Network loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"AI: Model loading error. Ensure the .pth file exists. {e}")
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
        # Ace Management
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

    # --- 3. BRAIN FOR BOTS ---
    def autoChoice(self, dealer_card_value):
        # If AI is not ready, use basic safety logic
        if not ai_ready:
            if self.count < 17: return 1
            return 0
            
        # Use the neural network to decide
        advice, _, _ = self.get_ai_prediction(dealer_card_value)
        if advice == "HIT":
            return 1
        else:
            return 0

    # --- 4. AI PREDICTION (Core Logic) ---
    def get_ai_prediction(self, dealer_val):
        """Prepares data and queries the neural network"""
        if not ai_ready:
            return None, 0, (100, 100, 100)

        # 1. Calculate if the hand is "Soft" (has an Ace usable as 11)
        raw_sum = sum(c.value for c in self.hand)
        is_soft = 1.0 if self.count > raw_sum else 0.0
        
        # 2. Prepare input tensor [PlayerTotal, DealerCard, Soft]
        # PyTorch expects float32
        input_tensor = torch.tensor([[float(self.count), float(dealer_val), is_soft]])

        # 3. Ask the model
        with torch.no_grad(): # Do not calculate gradients, we are in game
            win_prob, action_logits = ai_model(input_tensor)
        
        # 4. Interpret results
        # action_logits is [stand_value, hit_value]
        # Use Softmax to transform them into percentages
        probs = F.softmax(action_logits, dim=1)
        prob_stand = probs[0][0].item() * 100
        prob_hit = probs[0][1].item() * 100
        
        # Choose the best action
        if prob_hit > prob_stand:
            return "HIT", prob_hit, (50, 255, 50) # Green
        else:
            return "PASS", prob_stand, (255, 80, 80) # Red

    # --- BADGE DRAWING  ---
    def get_ai_advice(self, dealer_card):
        return self.get_ai_prediction(dealer_card.value)

    def draw_ai_badge(self, surface, mossa, prob, x, y, color):
        try:
            font_title = pygame.font.SysFont("Arial", 16, bold=True)
            font_sub = pygame.font.SysFont("Arial", 12) 
        except:
            font_title = pygame.font.Font(None, 16)
            font_sub = pygame.font.Font(None, 16)
            
        # Row 1: The Move
        text_mossa = font_title.render(f"NNLogic: {mossa}", True, (255, 255, 255))
        # Row 2: The Probability (in light gray for contrast)
        text_prob = font_sub.render(f"Win rate: {prob:.1f}%", True, (220, 220, 220))

        padding_x = 20
        padding_y = 15
        box_width = max(text_mossa.get_width(), text_prob.get_width()) + padding_x
        box_height = text_mossa.get_height() + text_prob.get_height() + padding_y
        
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
        
         # Center the first row at the top of the box
        rect_mossa = text_mossa.get_rect(centerx=box_rect.centerx, top=box_rect.top + 5)
        surface.blit(text_mossa, rect_mossa)

        # Center the second row immediately below the first
        rect_prob = text_prob.get_rect(centerx=box_rect.centerx, top=rect_mossa.bottom + 2)
        surface.blit(text_prob, rect_prob)

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

        name_color = GREEN if self.currentTurn else WHITE
        draw_text(surface, f"{self.name}   ${self.bank}", FONT_NORMAL, name_color,
                  self.x, self.y + card_h * 0.75)

        if self.currentTurn and self.is_human:
            draw_text(surface, "Hit(H) or Pass(P)", FONT_NORMAL, name_color,
                      self.x, self.y - card_h * 0.75)

            # AI Badge (Visible only to human)
            if dealer_card:
                mossa, prob, colore = self.get_ai_advice(dealer_card)
                if mossa:
                    self.draw_ai_badge(surface, mossa, prob, self.x, self.y - 130, colore)

        if self.bust:
            bust = load_image("Resources/Icons/bust.png")
            surface.blit(bust, (self.x - bust.get_width()/2, self.y - bust.get_height()/2))

        if self.blackjack:
            bj = load_image("Resources/Icons/blackjack.png")
            surface.blit(bj, (self.x - bj.get_width()/2, self.y - bj.get_height()/2))