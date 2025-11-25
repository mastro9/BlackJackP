import pygame
from utils.constants import WHITE, BLUE, FONT_NORMAL
from utils.helpers import draw_text, load_image
import sys


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

                    if event.key == pygame.K_h:
                        return 1

                    if event.key == pygame.K_p:
                        return 2

                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

    def addCard(self, card):
        self.hand.append(card)
        self.countCards()
    

    # this method considers all aces in a player's hand to give them the closest count under 21
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

    def drawHand(self, surface):
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
        """ Restituisce 1 = HIT, 0 = PASS """

        # Esempio: strategia di base semplificata
        # Se il bot ha <= 11, sempre HIT
        if self.count <= 11:
            return 1

        # Se dealer mostra una carta forte (7–A) e il bot ha < 17 → HIT
        if dealer_card_value >= 7 and self.count < 17:
            return 1

        # Se dealer ha carta debole (2–6) e il bot ha >= 12 → PASS
        if dealer_card_value <= 6 and self.count >= 12:
            return 0

        # Default: PASS sopra 16
        if self.count >= 17:
            return 0

        return 1
