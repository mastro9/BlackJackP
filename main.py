# all of our imports are listed here
import math
import random
import pygame
from pygame.locals import *
import sys


class Card:

    # this class contains all attributes of a playing card
    def __init__(self, suit, color, label, value):
        self.suit = suit
        self.color = color
        self.label = label
        self.value = value


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


class Dealer:

    # this class contains everything that is within the control of the dealer
    def __init__(self):
        self.deck = Deck()
        self.deck.createDeck()
        self.deck.shuffleDeck()
        self.hand = []
        self.count = 0
        self.x = halfWidth
        self.y = 100

    # this method creates the two-card hand that the dealer starts with
    def createDealerHand(self):
        for i in range(1, 3):
            self.addCard()

    # this method uses the getCard() to deal a card to a player
    def dealCard(self):
        return self.deck.getCard()

    # this method allows the dealer to deal himself a card and also account for the dealer's count
    def addCard(self):
        dealerCard = self.dealCard()
        self.hand.append(dealerCard)
        self.count += dealerCard.value
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

    # this method considers all aces in a dealer's hand to give them the closest count under 21
    def countAce(self):
        if self.count <= 21:
            for card in self.hand:
                if card.label == "A":
                    self.count += 10
                    if self.count > 21:
                        self.count -= 10
                        break

    # this method will draw all the cards in a hand (13 : 20 Card Dimension Ratio)
    def drawHand(self, surface):
        cardWidth, cardHeight = 78, 120
        cardGap = 20
        playerBoxLength, playerBoxHeight = cardWidth + (cardGap * (len(self.hand) - 1)), cardHeight
        playerTopLeftX = self.x - (0.5 * playerBoxLength)
        playerTopLeftY = self.y - (0.5 * playerBoxHeight)
        for card in self.hand:
            if card == self.hand[1]:
                drawCard = pygame.image.load("Resources/Cards/Back/red_back.png")
            else:
                drawCard = pygame.image.load("Resources/Cards/" + str(card.suit) + "/" + str(card.label) + ".png")
            resizedCard = pygame.transform.scale(drawCard, (cardWidth, cardHeight))
            surface.blit(resizedCard, (playerTopLeftX, playerTopLeftY))
            pygame.display.update()
            playerTopLeftX += cardGap
        nameX = self.x
        nameY = self.y + (0.75 * cardHeight)
        add_text("DEALER", text_Normal, surface, nameX, nameY, white)
        deckX = 400 - (0.5 * cardWidth)
        deckY = 100 - (0.5 * cardHeight)
        for card in range(1, 7):
            deckCard = pygame.image.load("Resources/Cards/Back/red_back.png")
            backCard = pygame.transform.scale(deckCard, (cardWidth, cardHeight))
            surface.blit(backCard, (deckX, deckY))
            deckX += cardGap


class Player:

    # this class contains everything that is within the control of the player
    def __init__(self, name):
        self.name = name
        self.hand = []
        self.count = 0
        self.blackjack = False
        self.bust = False
        self.bank = 100
        self.bet = 0
        self.roundsWon = 0
        self.x = 0
        self.y = 0
        self.currentTurn = False

    # this method asks the player for their choice of action when it is their turn
    def askChoice(self):
        inp = 0
        answered = False
        while answered is False:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN and event.key == K_h:
                    inp = 1
                    answered = True
                if event.type == KEYDOWN and event.key == K_p:
                    inp = 2
                    answered = True
        return inp

    # this method adds a card provided by the dealer to the player's hand
    def addCard(self, card):
        self.hand.append(card)
        self.countCards()
        print(str(self.name) + "'s Count: " + str(self.count))

    # this method prints the player's hand
    def printHand(self):
        print("")
        print(str(self.name) + "'s Hand: ")
        for playerCard in self.hand:
            print("Suit - " + playerCard.suit + "\nLabel - " + str(playerCard.label))

    # this method prints the player's count
    def printCount(self):
        print("")
        print(str(self.name) + "'s Count: " + str(self.count))

    # this method will apply the outcome of the bet to the bank
    # no negative parameter will ever need to passed as we have already subtracted the bet from the bank
    def applyBet(self, factor):
        self.bank += self.bet * factor

    # this method that will reset the bets
    def resetBet(self):
        self.bet = 0

    # this method prints the player's bank
    def printBank(self):
        print("")
        print(str(self.name) + "'s Bank: " + str(self.bank))

    # this method resets the player's hand
    def resetHandAndCount(self):
        self.hand = []
        self.count = 0

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

    # this method will draw all the cards in a hand (13 : 20 Card Dimension Ratio)
    def drawHand(self, surface):
        cardWidth, cardHeight = 78, 120
        cardGap = 20
        playerBoxLength, playerBoxHeight = cardWidth + (cardGap * (len(self.hand) - 1)), cardHeight
        playerTopLeftX = self.x - (0.5 * playerBoxLength)
        playerTopLeftY = self.y - (0.5 * playerBoxHeight)
        for card in self.hand:
            drawCard = pygame.image.load("Resources/Cards/" + str(card.suit) + "/" + str(card.label) + ".png")
            resizedCard = pygame.transform.scale(drawCard, (cardWidth, cardHeight))
            surface.blit(resizedCard, (playerTopLeftX, playerTopLeftY))
            pygame.display.update()
            playerTopLeftX += cardGap
        nameX = self.x
        nameY = self.y + (0.75 * cardHeight)
        nameColor = white
        if self.currentTurn:
            nameColor = blue
            add_text("Hit(H) or Pass(P)", text_Normal, surface, self.x, self.y - (0.75 * cardHeight), nameColor)
        if self.bust:
            bust = pygame.image.load("Resources/Icons/bust.png")
            bustWidth = bust.get_width()
            bustHeight = bust.get_height()
            surface.blit(bust, (self.x - (0.5 * bustWidth), self.y - (0.5 * bustHeight)))
        if self.blackjack:
            blackjack = pygame.image.load("Resources/Icons/blackjack.png")
            bjWidth = blackjack.get_width()
            bjHeight = blackjack.get_height()
            surface.blit(blackjack, (self.x - (0.5 * bjWidth), self.y - (0.5 * bjHeight)))
        add_text(str(self.name) + "   $" + str(self.bank), text_Normal, surface, nameX, nameY, nameColor)

    # function to reset everything for the next round
    def resetState(self):
        self.bust = False
        self.blackjack = False
        self.resetBet()
        self.resetHandAndCount()


# officially the end of all our class initializations and methods


# below consists the py-game/graphics related code

pygame.init()
pygame.font.init()
screenWidth, screenHeight = 1250, 750
halfWidth, halfHeight = screenWidth / 2, screenHeight / 2
pokerBackgroundOriginal = pygame.image.load("Resources/Icons/pokerBackground3.jpg")
pokerGreen = pygame.transform.scale(pokerBackgroundOriginal, (screenWidth, screenHeight))
black, blue, white, orange, red = (0, 0, 0), (51, 235, 255), (255, 255, 255), (255, 165, 0), (255, 0, 0)
fontType = 'Comic Sans MS'
text_Title = pygame.font.SysFont(fontType, 80)
text_Heading = pygame.font.SysFont(fontType, 60)
text_SubHeading = pygame.font.SysFont(fontType, 45)
text_Bold = pygame.font.SysFont(fontType, 30)
text_Normal = pygame.font.SysFont(fontType, 20)
text_Small = pygame.font.SysFont(fontType, 10)

# global variables listed below
players = []
numPlayers = 0
dealer = Dealer()
startY = 50


# function to add text to the game when needed
def add_text(text, font, surface, x, y, text_color):
    textObject = font.render(text, False, text_color)
    textWidth = textObject.get_rect().width
    textHeight = textObject.get_rect().height
    surface.blit(textObject, (x - (textWidth / 2), y - (textHeight / 2)))

# function to create the start screen for the game
def startGame():
    pygame.init()
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    pygame.display.set_caption("Welcome")
    screen.blit(pokerGreen, (0, 0))
    titleLogo = pygame.image.load("Resources/Icons/titleBlitzEdition.png")
    logoX = titleLogo.get_width()
    logoY = titleLogo.get_height()
    screen.blit(titleLogo, (halfWidth - (0.5 * logoX), halfHeight - (0.5 * logoY) - 25))
    add_text("PRESS SPACE TO CONTINUE", text_SubHeading, screen, halfWidth, halfHeight + 100, white)
    pygame.display.update()
    beginning = True
    while beginning:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key == K_SPACE:
                beginning = False

# function to show the reader all of the instructions and rules to our game
def showInstructions():
    pygame.init()
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    pygame.display.set_caption("How to Play")
    screen.blit(pokerGreen, (0, 0))
    add_text("Goal of the Game:", text_SubHeading, screen, halfWidth, 50, orange)
    add_text("--> To get the closest to 21 without going over in order for you to make money.", text_Normal, screen, halfWidth, 100, white)
    add_text("Basic Rules:", text_SubHeading, screen, halfWidth, 150, orange)
    add_text("--> Cards 2 - 10 = face value        Jack, Queen, King = 10        Ace = 1 or 11", text_Normal, screen, halfWidth, 200, white)
    add_text("--> Press H to Hit (Gets a card)        Press P to Pass (Finishes turn)", text_Normal, screen, halfWidth, 250, white)
    add_text("--> You may hit as much as you want, however, once you pass 21, you bust and your turn is over.", text_Normal, screen, halfWidth, 300, white)
    add_text("--> If you get to 21 with your first two cards, you blackjack, and you sit out for that round.", text_Normal, screen, halfWidth, 350, white)
    add_text("Betting:", text_SubHeading, screen, halfWidth, 400, orange)
    add_text("--> Everyone has $100 to start the game.", text_Normal, screen, halfWidth, 450, white)
    add_text("--> Be careful, because you can go bankrupt, however the game will always supply you with at least a dollar to play.", text_Normal, screen, halfWidth, 500, white)
    add_text("--> Bust = Dealer takes your bet        Blackjack = Earn 1 and a half times your bet", text_Normal, screen, halfWidth, 550, white)
    add_text("--> Closest to 21 = Earn 2 times your bet, else Dealer takes your bet", text_Normal, screen, halfWidth, 600, white)
    add_text("--> If your count is equal to the dealer's and it is the highest count under 21, you get your bet back", text_Normal, screen, halfWidth, 650, white)
    add_text("--> Dealer Bust = Everyone remaining in the game earns 2 times their bets.", text_Normal, screen, halfWidth, 700, white)
    pygame.display.update()
    instructions = True
    while instructions:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key == K_SPACE:
                instructions = False

# function to receive the number of players
def getNumberOfPlayers():
    global numPlayers
    pygame.init()
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    pygame.display.set_caption("Enter Players")
    validNumbers = "23456"
    userString = ""
    answered = False
    while answered is False:
        screen.blit(pokerGreen, (0, 0))
        add_text("Enter the number of players playing below (Game is designed for 2-6 players):", text_Bold, screen, halfWidth, halfHeight - 50, orange)
        add_text(userString, text_SubHeading, screen, halfWidth, halfHeight, white)
        add_text("PRESS SPACE TO CONTINUE", text_Bold, screen, halfWidth, halfHeight + 50, orange)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (pygame.key.name(event.key) in validNumbers) and len(userString) < 1:
                userString += str(pygame.key.name(event.key))
            if event.type == KEYDOWN and event.key == K_BACKSPACE:
                userString = ""
            if event.type == KEYDOWN and event.key == K_SPACE and len(userString) == 1:
                numPlayers = int(userString)
                answered = True

# function to receive the names of all the players
def getPlayerNames():
    global players, numPlayers
    pygame.init()
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    pygame.display.set_caption("Enter Names")
    validCharacters = "abcdefghijklmnopqrstuvwxyz1234567890"
    allNames = False
    while allNames is False:
        for player in range(1, numPlayers + 1):
            userString = ""
            singleName = False
            while singleName is False:
                screen.blit(pokerGreen, (0, 0))
                add_text("Enter player " + str(player) + "'s name:", text_Bold, screen, halfWidth, halfHeight - 50, orange)
                add_text(userString, text_SubHeading, screen, halfWidth, halfHeight, white)
                if player < numPlayers:
                    add_text("PRESS SPACE TO ADD NAME", text_Bold, screen, halfWidth, halfHeight + 50, orange)
                elif player == numPlayers:
                    add_text("PRESS SPACE TO CONTINUE", text_Bold, screen, halfWidth, halfHeight + 50, orange)
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN and (pygame.key.name(event.key) in validCharacters) and len(userString) < 9:
                        userString += str(pygame.key.name(event.key))
                    if event.type == KEYDOWN and event.key == K_BACKSPACE:
                        userString = ""
                    if event.type == KEYDOWN and event.key == K_SPACE:
                        players.append(Player(userString))
                        singleName = True
                        if player == numPlayers:
                            allNames = True

# function that sets the x and y coordinates of every player plus the dealer
def fixCoordinates():
    global players, dealer, numPlayers
    if numPlayers == 1:
        players[0].x = halfWidth
        players[0].y = 650
    elif numPlayers == 2:
        players[0].x = 850
        players[0].y = halfHeight + 150
        players[1].x = 400
        players[1].y = halfHeight + 150
    elif numPlayers == 3:
        players[0].x = 1000
        players[0].y = halfHeight
        players[1].x = halfWidth
        players[1].y = 625
        players[2].x = 250
        players[2].y = halfHeight
    elif numPlayers == 4:
        players[0].x = 1050
        players[0].y = halfHeight
        players[1].x = halfWidth + 200
        players[1].y = 625
        players[2].x = halfWidth - 200
        players[2].y = 625
        players[3].x = 200
        players[3].y = halfHeight
    elif numPlayers == 5:
        players[0].x = 1100
        players[0].y = halfHeight - 50
        players[1].x = halfWidth + 300
        players[1].y = 525
        players[2].x = halfWidth
        players[2].y = 625
        players[3].x = halfWidth - 300
        players[3].y = 525
        players[4].x = 150
        players[4].y = halfHeight - 50
    elif numPlayers == 6:
        players[0].x = 1100
        players[0].y = halfHeight - 170
        players[1].x = halfWidth + 350
        players[1].y = 415
        players[2].x = halfWidth + 175
        players[2].y = 625
        players[3].x = halfWidth - 175
        players[3].y = 625
        players[4].x = halfWidth - 350
        players[4].y = 415
        players[5].x = 150
        players[5].y = halfHeight - 170

# function to collect the bets of all the players every round
def getPlayerBets():
    # for player in players:
    #     player.createBet()
    # for player in players:
    #     player.bet = 5
    #     player.applyBet(-1)
    global players, numPlayers
    pygame.init()
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    pygame.display.set_caption("Enter Bets")
    validNumbers = "1234567890"
    allBets = False
    while allBets is False:
        for player in players:
            if player.bank == 0:
                player.bank += 1
            validBet = True
            userString = ""
            singleBet = False
            while singleBet is False:
                screen.blit(pokerGreen, (0, 0))
                add_text("Enter " + str(player.name) + "'s bet (" + str(player.name) + "'s Bank = $" + str(player.bank) + "):", text_Bold, screen, halfWidth, halfHeight - 50,
                         orange)
                add_text(userString, text_SubHeading, screen, halfWidth, halfHeight, white)
                if player is not players[len(players) - 1]:
                    add_text("PRESS SPACE TO CONTINUE", text_Bold, screen, halfWidth, halfHeight + 50, orange)
                else:
                    add_text("PRESS SPACE TO START GAME", text_Bold, screen, halfWidth, halfHeight + 50, orange)
                if validBet is False:
                    add_text("ENTER A VALID BET", text_Bold, screen, halfWidth, halfHeight + 100, red)
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN and (pygame.key.name(event.key) in validNumbers) and len(
                            userString) < 4:
                        userString += str(pygame.key.name(event.key))
                    if event.type == KEYDOWN and event.key == K_BACKSPACE:
                        userString = ""
                    if event.type == KEYDOWN and event.key == K_SPACE:
                        if userString == "":
                            userString = "0"
                        if 0 <= int(userString) <= player.bank:
                            singleBet = True
                            player.bet = int(userString)
                            player.applyBet(-1)
                        if int(userString) > player.bank:
                            validBet = False
                        if singleBet is True and player == players[len(players) - 1]:
                            allBets = True

# function to restore all the cards to the deck for a new round
def newDeck():
    global dealer
    dealer = Dealer()

# function to create the hands of the dealer and all the players
def createHands():
    global dealer
    dealer.createDealerHand()
    for i in range(1, 3):
        for player in players:
            card = dealer.dealCard()
            player.addCard(card)

# function to check for blackjacks (basically when the first two initial cards dealt to a player add up to 21)
# if there is one you automatically win one and a half times your bet, and you sit out for the round
def checkBlackJack():
    for player in players:
        if player.count == 21:
            print("")
            print(player.name + ", you got a BLACKJACK & won one and a half times your bet.")
            player.applyBet(3/2)
            player.resetBet()
            player.blackjack = True

# function to run the turns of all players, basically allowing to hit and pass as normal
# this function also contains the code for changing the value of an ace when necessary
def playTurns():
    pygame.init()
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    pygame.display.set_caption("Play Round")
    turn = 0
    while turn < len(players):
        currentPlayer = players[turn]
        if currentPlayer.blackjack:
            turn += 1
            if turn >= len(players):
                break
            currentPlayer = players[turn]
        for player in players:
            player.currentTurn = False
            if player == currentPlayer:
                player.currentTurn = True
        currentPlayer.printHand()
        drawTurn(screen)
        choice = currentPlayer.askChoice()
        if choice == 1:
            keepHitting = True
            while keepHitting is True:
                hitCard = dealer.dealCard()
                currentPlayer.addCard(hitCard)
                drawTurn(screen)
                currentPlayer.printHand()
                if currentPlayer.count > 21:
                    print("")
                    print(str(currentPlayer.name) + ", you busted. The Dealer gets your bet.")
                    currentPlayer.bust = True
                    currentPlayer.resetBet()
                    break
                choice = currentPlayer.askChoice()
                if choice != 1:
                    keepHitting = False
        turn += 1

# function to draw the screen every time an action is conducted in the playing of the game
def drawTurn(surface):
    global players, dealer
    surface.blit(pokerGreen, (0, 0))
    dealer.drawHand(surface)
    for player in players:
        player.drawHand(surface)
    tigerEye = pygame.image.load("Resources/Icons/tigerEye.png")
    tigerX = round(tigerEye.get_width() * 0.5)
    tigerY = round(tigerEye.get_height() * 0.5)
    tigerEye = pygame.transform.scale(tigerEye, (tigerX, tigerY))
    tigerEye.set_alpha(10)
    surface.blit(tigerEye, (halfWidth - (0.5 * tigerX), halfHeight - (0.5 * tigerY)))
    pygame.display.update()

# function to reveal the face down card of the dealer, and if the dealer has to force hit he will do so
def revealDealerHand(surface):
    global dealer, startY
    dealerBust = False
    while dealer.count <= 16:
        dealer.addCard()
    if dealer.count > 21:
        print("")
        print("The Dealer busted. You all got double your bets.")
        startY += 50
        add_text("The Dealer busted. You all got double your bets.", text_Normal, surface, halfWidth, startY, white)
        for player in players:
            if player.bet > 0:
                player.applyBet(2)
        dealerBust = True
    dealer.printDealerHand()
    dealer.printDealerCount()
    return dealerBust

# function to see how the bets are retrieved based on counts
def compareCounts(surface):
    global players, dealer, startY
    noCounts = True
    highestCount = 0
    for player in players:
        if 21 >= player.count > highestCount:
            highestCount = player.count
            noCounts = False
    if noCounts is False:
        for player in players:
            if player.count == highestCount and highestCount > dealer.count and player.blackjack is False:
                print("")
                print(str(player.name) + ", you won twice your bet")
                startY += 50
                add_text(str(player.name) + ", you won twice your bet.", text_Normal, surface, halfWidth, startY, white)
                player.applyBet(2)
                player.resetBet()
            elif player.count == dealer.count and player.blackjack is False:
                print("")
                print(str(player.name) + ", you got your bet back.")
                startY += 50
                add_text(str(player.name) + ", you got your bet back.", text_Normal, surface, halfWidth, startY, white)
                player.applyBet(1)
                player.resetBet()
            elif player.count < dealer.count and player.blackjack is False:
                print("")
                print(str(player.name) + ", the dealer took your bet.")
                startY += 50
                add_text(str(player.name) + ", the dealer took your bet.", text_Normal, surface, halfWidth, startY, white)
                player.resetBet()
            elif player.bust:
                startY += 50
                add_text(str(player.name) + " busted.", text_Normal, surface, halfWidth, startY, white)
            elif player.blackjack:
                startY += 50
                add_text(str(player.name) + " got a blackjack.", text_Normal, surface, halfWidth, startY, white)
    else:
        startY += 50
        add_text("You all busted.", text_Normal, surface, halfWidth, startY, white)

# function to check for a winner, basically when a person reaches a certain target amount of money
def checkWinner(surface):
    global roundOver, gameOver, startY
    highestBank = 0
    winnerPresent = False
    for player in players:
        if player.bank > highestBank and player.bank >= 200:
            highestBank = player.bank
            winnerPresent = True
    if winnerPresent:
        for player in players:
            if player.bank == highestBank:
                print("")
                print(str(player.name) + ", YOU WON THE GAME.")
                startY += 50
                add_text(str(player.name) + ", YOU WON THE GAME.", text_Normal, surface, halfWidth, startY, blue)
            roundOver = True
        gameOver = True

# function to display a message about the results of the previous round
def showEndRoundScreen():
    global startY, gameOver, numPlayers, players
    pygame.init()
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    pygame.display.set_caption("Round Over")
    screen.blit(pokerGreen, (0, 0))
    add_text("Results:", text_SubHeading, screen, halfWidth, startY, orange)
    if revealDealerHand(screen) is False:
        compareCounts(screen)
    checkWinner(screen)
    startY += 50
    add_text("Dealer's Count: " + str(dealer.count), text_Normal, screen, halfWidth, startY, orange)
    startY += 50
    countString1 = ""
    for i in range(0, 3):
        player = players[i]
        if (numPlayers > 3 and player == players[2]) or player == players[numPlayers - 1]:
            countString1 += str(player.name) + "'s Count: " + str(player.count)
        else:
            countString1 += str(player.name) + "'s Count: " + str(player.count) + "        "
        if i == 1 and numPlayers == 2:
            break
    add_text(countString1, text_Normal, screen, halfWidth, startY, orange)
    if numPlayers > 3:
        startY += 50
        countString2 = ""
        for i in range(3, numPlayers):
            player = players[i]
            if player == players[numPlayers - 1]:
                countString2 += str(player.name) + "'s Count: " + str(player.count)
            else:
                countString2 += str(player.name) + "'s Count: " + str(player.count) + "        "
        add_text(countString2, text_Normal, screen, halfWidth, startY, orange)
    bankString1 = ""
    for i in range(0, 3):
        player = players[i]
        if (numPlayers > 3 and player == players[2]) or player == players[numPlayers - 1]:
            bankString1 += str(player.name) + "'s Bank: $" + str(player.bank)
        else:
            bankString1 += str(player.name) + "'s Bank: $" + str(player.bank) + "        "
        if i == 1 and numPlayers == 2:
            break
    add_text(bankString1, text_Normal, screen, halfWidth, 600, white)
    if numPlayers > 3:
        bankString2 = ""
        for i in range(3, numPlayers):
            player = players[i]
            if player == players[numPlayers - 1]:
                bankString2 += str(player.name) + "'s Bank: $" + str(player.bank)
            else:
                bankString2 += str(player.name) + "'s Bank: $" + str(player.bank) + "        "
        add_text(bankString2, text_Normal, screen, halfWidth, 650, white)
    if gameOver is True:
        add_text("PRESS SPACE TO EXIT", text_SubHeading, screen, halfWidth, 700, orange)
    else:
        add_text("PRESS SPACE TO CONTINUE", text_SubHeading, screen, halfWidth, 700, orange)
    pygame.display.update()
    roundEnd = True
    while roundEnd:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key == K_SPACE:
                roundEnd = False

# function to reset things such as the bets, and player hands for a new round (plus we need to reset the starting Y
# value for all the text shown in the end of the round
def resetStats():
    global players, startY
    for player in players:
        player.printBank()
        player.resetState()
    startY = 100


# main game loop starts here

gameOver = False
while gameOver is False:
    startGame()
    showInstructions()
    getNumberOfPlayers()
    getPlayerNames()
    fixCoordinates()
    roundOver = False
    while roundOver is False:
        getPlayerBets()
        newDeck()
        createHands()
        checkBlackJack()
        playTurns()
        showEndRoundScreen()
        resetStats()
