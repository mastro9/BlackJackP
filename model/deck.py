from .card import Card
import random

class Deck:

    # this class contains an array that acts as our 52 card deck
    def __init__(self):
        self.cards = []

    # this method simply creates a deck using the Card class above
    def createDeck(self):
        suits = ["Clover", "Spade", "Heart", "Diamond"]
        for symbol in suits:
            number = 2
            while number < 15:
                if symbol == "Clover" or symbol == "Spade":
                    suitColor = "Black"
                else:
                    suitColor = "Red"
                value = number
                if number > 10:
                    value = 10
                if number == 14:
                    value = 1
                letter = number
                if number == 11:
                    letter = "J"
                elif number == 12:
                    letter = "Q"
                elif number == 13:
                    letter = "K"
                elif number == 14:
                    letter = "A"
                newCard = Card(symbol, suitColor, letter, value)
                self.cards.append(newCard)
                number += 1
                
    # this method allows us to shuffle our deck so that it is randomly arranged
    def shuffleDeck(self):
        return random.shuffle(self.cards)

    # this method basically gets the top card of the deck and returns it
    def getCard(self):
        topCard = self.cards[0]
        self.cards.pop(0)
        return topCard