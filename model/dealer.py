import pygame
from utils.constants import HALF_WIDTH, WHITE, FONT_NORMAL
from utils.helpers import draw_text, load_image
from .deck import Deck


class Dealer:
    def __init__(self):
        self.deck = Deck()
        self.deck.createDeck()
        self.deck.shuffleDeck()
        self.hand = []
        self.count = 0
        self.x = HALF_WIDTH
        self.y = 100
        self.revealed = False

    def createDealerHand(self):
        for _ in range(2):
            self.addCard()

    def dealCard(self):
        return self.deck.getCard()

    def addCard(self):
        card = self.dealCard()
        self.hand.append(card)
        self.count += card.value
        self.countAce()

    # this method prints the dealer's hand
    def printDealerHand(self):
        print("")
        print("Dealer's Hand: ")
        for dealerCard in self.hand:
            print("Suit: " + dealerCard.suit + "\nLabel: " + str(dealerCard.label))

    # this method prints the dealer's count
    def printDealerCount(self):
        print("")
        print("Dealer's Count: " + str(self.count))


    def countAce(self):
        if self.count <= 21:
            for card in self.hand:
                if card.label == "A":
                    self.count += 10
                    if self.count > 21:
                        self.count -= 10
                        break

    def drawHand(self, surface):
        card_w, card_h = 78, 120
        gap = 20

        start_x = self.x - (card_w + gap * (len(self.hand)-1)) / 2
        start_y = self.y - card_h / 2

        for i, card in enumerate(self.hand):

            # Se la carta è la seconda e NON è ancora rivelata →
            if i == 1 and not self.revealed:
                img = load_image("Resources/Cards/Back/red_back.png", (card_w, card_h))
            else:
                path = f"Resources/Cards/{card.suit}/{card.label}.png"
                img = load_image(path, (card_w, card_h))

            surface.blit(img, (start_x, start_y))
            start_x += gap

        draw_text(surface, "DEALER", FONT_NORMAL, WHITE, self.x, self.y + card_h * 0.75)

        # disegna il mazzo
        deck_x = 400 - card_w/2
        deck_y = 100 - card_h/2

        for _ in range(6):
            back = load_image("Resources/Cards/Back/red_back.png", (card_w, card_h))
            surface.blit(back, (deck_x, deck_y))
            deck_x += gap
    
    def reveal_all_cards(self):
        """Segna che tutte le carte del dealer devono essere mostrate (nessuna coperta)."""
        self.revealed = True