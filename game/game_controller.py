import pygame
import random
import sys # Needed for sys.exit() in the functions at the bottom

# Unmodified imports
from model.playerNN import Player
from model.dealer import Dealer
from game.game_logic import (
    deal_initial_cards,
    check_blackjack,
    play_turns,
    resolve_round,
)
from utils.constants import *
from utils.helpers import *


class GameController:

    def __init__(self):
        pygame.init()
        self.players = []
        self.dealer = Dealer()
        self.game_over = False

    def start(self):
        show_start_screen()
        show_instructions()

        num = 4
        self.players = [] # reset list

        # --- HUMAN PLAYER ---
        # *** Important: get_player_name() must return a Player object, not a list ***
        human = get_player_name() 
        self.players.append(human)
        human.is_human = True

        # --- OTHER 3 BOTS ---
        for i in range(1, 4):
            # Assuming Player starts with a default bankroll > 0
            bot = Player(f"Player{i}") 
            self.players.append(bot)
        
        self.fix_coordinates(num)

        while not self.game_over:
            
            # 1. CHECK AND REMOVE BANKRUPT PLAYERS
            self.remove_bankrupt_players()
            
            # If only one player (or none) remains after the check, the game ends.
            # If the list is empty, we exit the loop.
            if len(self.players) < 1:
                print("All players are bankrupt. Game Over.")
                self.game_over = True
                continue # Move to the next cycle (which will exit)

            # 2. BETTING MANAGEMENT
            # The get_bet function handles the human player's bet (self.players[0])
            # Note: get_bet must handle the case where the human player has 0 money
            get_bet(self.players) 
            
            # BOT Bets
            # Iterate over the updated list, excluding the human player (who already bet)
            for p in self.players[1:]:
                # ADDITIONAL SAFETY CHECK FOR BOTS (p.bank should always be > 0 here)
                if p.bank <= 0:
                    continue # skip the bot if bankrupt (even if it was removed)

                bet = random.randint(1, min(100, int(p.bank)))
                p.bet = bet
                p.bank -= bet

            # 3. ROUND LOGIC
            # new deck
            self.dealer = Dealer()

            deal_initial_cards(self.players, self.dealer)
            check_blackjack(self.players)
            play_turns(self.players, self.dealer)

            resolve_round(self.players, self.dealer)
            
            # Pass a flag to show the correct exit message if the game is over
            show_end_round(self.players, self.dealer, game_over=self.game_over or len(self.players) < 1)

            # 4. PREPARING NEW ROUND
            self.reset_for_new_round()
            self.check_final_winner()
            
    def remove_bankrupt_players(self):
        """Removes players who have a bankroll of 0 or less."""
        active_players = []
        bankrupt_players = []
        
        for p in self.players:
            if p.bank <= 0:
                bankrupt_players.append(p)
                print(f"💰 {p.name} is bankrupt (${p.bank}) and has been removed from the game.")
            else:
                active_players.append(p)
                
        self.players = active_players
        
        # After removal, we might need to reassign coordinates to remaining players
        self.fix_coordinates(len(self.players))


    def fix_coordinates(self, num):
        players = self.players
        if num == 1:
            if players:
                players[0].x = HALF_WIDTH
                players[0].y = 650

        elif num == 2:
            players[0].x = 850
            players[0].y = HALF_HEIGHT + 150
            players[1].x = 400
            players[1].y = HALF_HEIGHT + 150

        elif num == 3:
            players[0].x = 1000
            players[0].y = HALF_HEIGHT
            players[1].x = HALF_WIDTH
            players[1].y = 625
            players[2].x = 250
            players[2].y = HALF_HEIGHT

        elif num == 4:
            players[0].x = 1050
            players[0].y = HALF_HEIGHT
            players[1].x = HALF_WIDTH + 200
            players[1].y = 625
            players[2].x = HALF_WIDTH - 200
            players[2].y = 625
            players[3].x = 200
            players[3].y = HALF_HEIGHT

        elif num == 5:
            players[0].x = 1100
            players[0].y = HALF_HEIGHT - 50
            players[1].x = HALF_WIDTH + 300
            players[1].y = 525
            players[2].x = HALF_WIDTH
            players[2].y = 625
            players[3].x = HALF_WIDTH - 300
            players[3].y = 525
            players[4].x = 150
            players[4].y = HALF_HEIGHT - 50

        elif num == 6:
            players[0].x = 1100
            players[0].y = HALF_HEIGHT - 170
            players[1].x = HALF_WIDTH + 350
            players[1].y = 415
            players[2].x = HALF_WIDTH + 175
            players[2].y = 625
            players[3].x = HALF_WIDTH - 175
            players[3].y = 625
            players[4].x = HALF_WIDTH - 350
            players[4].y = 415
            players[5].x = 150
            players[5].y = HALF_HEIGHT - 170
        

    def reset_for_new_round(self):
        for p in self.players:
            p.resetState()

    # win with $200
    def check_final_winner(self):
        for p in self.players:
            if p.bank >= 200:
                print(f"🎉 {p.name} won!")
                self.game_over = True
                
def show_start_screen():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Welcome")

    # background
    screen.blit(POKER_BACKGROUND, (0, 0))

    # Load the logo
    title_logo = pygame.image.load("Resources/Icons/logo.png")

    # Define new dimensions 
    new_width = title_logo.get_width() // 2
    new_height = title_logo.get_height() // 2

    # Scale the image 
    title_logo = pygame.transform.smoothscale(title_logo, (new_width, new_height))

    # Update dimension variables for centering calculation
    logo_w = title_logo.get_width()
    logo_h = title_logo.get_height()

    # Draw the logo
    screen.blit(title_logo, (HALF_WIDTH - logo_w // 2, HALF_HEIGHT - logo_h // 2 - 25))

    # "press space" text
    draw_text(screen,"PRESS SPACE TO CONTINUE",FONT_SUBTITLE,WHITE,HALF_WIDTH,HALF_HEIGHT + 100)

    pygame.display.update()
    wait_for_input()

def show_instructions():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("How to Play")
    screen.blit(POKER_BACKGROUND, (0, 0))

    # TITLES AND RULES — ORIGINAL STYLE
    draw_text(screen, "Goal of the Game:", FONT_SUBTITLE, ORANGE, HALF_WIDTH, 50)
    draw_text(screen,
              "--> To get the closest to 21 without going over in order for you to make money.",
              FONT_NORMAL, WHITE, HALF_WIDTH, 100)

    draw_text(screen, "Basic Rules:", FONT_SUBTITLE, ORANGE, HALF_WIDTH, 150)
    draw_text(screen,
              "--> Cards 2 - 10 = face value    Jack, Queen, King = 10    Ace = 1 or 11",
              FONT_NORMAL, WHITE, HALF_WIDTH, 200)
    draw_text(screen,
              "--> Press H to Hit (Gets a card)        Press P to Pass (Finishes turn)",
              FONT_NORMAL, WHITE, HALF_WIDTH, 250)
    draw_text(screen,
              "--> You may hit as much as you want, however, once you pass 21, you bust.",
              FONT_NORMAL, WHITE, HALF_WIDTH, 300)
    draw_text(screen,
              "--> If you get to 21 with your first two cards, you blackjack and sit out.",
              FONT_NORMAL, WHITE, HALF_WIDTH, 350)

    draw_text(screen, "Betting:", FONT_SUBTITLE, ORANGE, HALF_WIDTH, 400)
    draw_text(screen,
              "--> Everyone has $100 to start the game.",
              FONT_NORMAL, WHITE, HALF_WIDTH, 450)
    draw_text(screen,
              "--> Bust = Dealer takes your bet    Blackjack = Earn 1.5x your bet",
              FONT_NORMAL, WHITE, HALF_WIDTH, 500)
    draw_text(screen,
              "--> Closest to 21 = Earn 2x your bet, else the dealer wins.",
              FONT_NORMAL, WHITE, HALF_WIDTH, 550)
    draw_text(screen,
              "--> Tie with dealer (if highest under 21) = Get your bet back",
              FONT_NORMAL, WHITE, HALF_WIDTH, 600)
    draw_text(screen,
              "--> Dealer Bust = Everyone still in the game earns 2x their bets.",
              FONT_NORMAL, WHITE, HALF_WIDTH, 650)

    pygame.display.update()
    wait_for_input()

def get_player_name():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Enter Names")

    valid_chars = "abcdefghijklmnopqrstuvwxyz1234567890"


    name = ""
    name_confirmed = False

    while not name_confirmed:
        # BACKGROUND
        screen.blit(POKER_BACKGROUND, (0, 0))

        draw_text(screen,f"Enter your name:",FONT_BOLD,ORANGE,HALF_WIDTH,HALF_HEIGHT - 50)
        # ENTERED NAME
        draw_text(screen,name,FONT_BOLD,WHITE,HALF_WIDTH,HALF_HEIGHT)

        footer = "PRESS SPACE TO CONTINUE"

        draw_text(screen,footer,FONT_BOLD,ORANGE,HALF_WIDTH,HALF_HEIGHT + 50)
        pygame.display.update()

        # EVENT HANDLING
        for event in pygame.event.get():

            # WINDOW CLOSING
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # ESC = EXIT
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

            # CHARACTER INSERTION
            if event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)

                # add character if valid and length < 9
                if key in valid_chars and len(name) < 9:
                        name += key

                # delete all
                elif event.key == pygame.K_BACKSPACE:
                    name = ""

                # confirm name with SPACE
                elif event.key == pygame.K_SPACE and len(name) > 0:
                    name_confirmed = True

    return Player(name)


def get_bet(players):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Enter Bets")

    valid_nums = "0123456789"
    
    # Takes the first player, assumed to be the human.
    # If the list is empty here, an IndexError will occur, but it is handled in GameController.start().
    p = players[0] 

    bet = ""
    bet_value = None
    valid_bet = True

    if p.bank <= 0:
        print(f"WARNING: {p.name} is bankrupt.")
        pass 

    while True:
        screen.blit(POKER_BACKGROUND, (0, 0))

        draw_text(screen, f"Enter {p.name}'s bet  ({p.name}'s Bank = ${p.bank})",
                          FONT_BOLD, ORANGE, HALF_WIDTH, HALF_HEIGHT - 50)
        draw_text(screen, bet, FONT_BOLD, WHITE, HALF_WIDTH, HALF_HEIGHT)

        footer = ("PRESS SPACE TO START GAME")
        draw_text(screen, footer, FONT_BOLD, ORANGE, HALF_WIDTH, HALF_HEIGHT + 50)

        if not valid_bet:
            # Message for invalid bet (e.g., too high or zero)
            draw_text(screen, "INVALID BET (Max $100 and Max Bankroll)",
                              FONT_BOLD, RED, HALF_WIDTH, HALF_HEIGHT + 100)

        pygame.display.update()

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)

                # numbers
                if key in valid_nums and len(bet) < 4:
                    bet += key
                    valid_bet = True
                    bet_value = None

                # delete
                elif event.key == pygame.K_BACKSPACE:
                    bet = ""
                    valid_bet = True

                # confirm
                elif event.key == pygame.K_SPACE:

                    # DO NOT allow empty bet
                    if bet == "":
                            valid_bet = False
                            continue

                    bet_value = int(bet)
                    
                    # Max bet is 100, and cannot exceed bankroll.
                    max_allowed_bet = min(100, p.bank)

                    # minimum 1, maximum max_allowed_bet
                    if 1 <= bet_value <= max_allowed_bet:
                        p.bet = bet_value
                        p.bank -= bet_value
                        valid_bet = True
                        break
                    else:
                        # The bet is invalid (e.g. 0, > 100, or > bank)
                        valid_bet = False
                        bet_value = None

        if bet_value is not None and valid_bet:
            break
            
def show_end_round(players, dealer, game_over=False):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Round Over")
    screen.blit(POKER_BACKGROUND_GAME, (0, 0))

    y = 100

    # TITLE
    draw_text(screen, "Results:", FONT_SUBTITLE, ORANGE, HALF_WIDTH, y)
    y += 60

    # DEALER
    draw_text(screen, f"Dealer's Count: {dealer.count}", FONT_NORMAL, ORANGE, HALF_WIDTH, y)
    y += 50

    num = len(players)

    # ---- ROW 1: Max 3 players ----
    row1 = ""
    for i in range(min(3, num)):
        p = players[i]
        part = f"{p.name}'s Count: {p.count}"
        if p == players[num-1]: 
            part = f"{p.name}'s Count: {p.count}" 
        else:
            part = f"{p.name}'s Count: {p.count}" + "   "
        row1 += part

    draw_text(screen, row1, FONT_NORMAL, ORANGE, HALF_WIDTH, y)

    # ---- ROW 2: only if there are more than 3 players ----
    if num > 3:
        y += 50
        row2 = ""
        for i in range(3, num):
            p = players[i]
            if i < num - 1:
                part = f"{p.name}'s Count: {p.count}" 
            else:
                part = f"{p.name}'s Count: {p.count}" + "        "
            row2 += part

        draw_text(screen, row2, FONT_NORMAL, ORANGE, HALF_WIDTH, y)

    # ---- BANK ROW 1 ----
    y = 600
    bank1 = ""
    for i in range(min(3, num)):
        p = players[i]
        if p == players[num-1]:
            part = f"{p.name}'s Bank: ${p.bank}" 
        else: 
            part = f"{p.name}'s Bank: ${p.bank}" + "        "
        bank1 += part

    draw_text(screen, bank1, FONT_NORMAL, WHITE, HALF_WIDTH, y)

    # ---- BANK ROW 2 ----
    if num > 3:
        y += 50
        bank2 = ""
        for i in range(3, num):
            p = players[i]
            part = f"{p.name}'s Bank: ${p.bank}"
            if i < num - 1:
                part += "        "
            bank2 += part

        draw_text(screen, bank2, FONT_NORMAL, WHITE, HALF_WIDTH, y)

    # FINAL MESSAGE
    footer = "PRESS SPACE TO EXIT" if game_over or num < 1 else "PRESS SPACE TO CONTINUE"
    draw_text(screen, footer, FONT_SUBTITLE, ORANGE, HALF_WIDTH, SCREEN_HEIGHT - 50)

    pygame.display.update()

    # WAIT FOR INPUT
    while True:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                if event.key == pygame.K_SPACE:
                    return
                
#def get_number_of_players():
#    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
#    pygame.display.set_caption("Enter Players")
#    user_input = ""
#    valid = "23456"

#    while True:
        # SFONDO
#        screen.blit(POKER_BACKGROUND, (0, 0))
#        draw_text(screen,"Enter the number of players (Game is designed for 2-6 players):",FONT_BOLD,ORANGE,HALF_WIDTH,HALF_HEIGHT - 50)
#        draw_text(screen,user_input,FONT_SUBTITLE,WHITE,HALF_WIDTH,HALF_HEIGHT)
#        draw_text(screen,"PRESS SPACE TO CONTINUE",FONT_BOLD,ORANGE,HALF_WIDTH,HALF_HEIGHT + 50)
#        pygame.display.update()

#        for event in pygame.event.get():
            #chiudi finestra
#            if event.type == pygame.QUIT:
#                pygame.quit()
#                sys.exit()
            #esc per chiudere finestra
#            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
#                pygame.quit()
#                sys.exit()
#            if event.type == pygame.KEYDOWN:
#                key = pygame.key.name(event.key)

#                if key in valid and len(user_input) == 0:
#                    user_input = key

#                elif event.key == pygame.K_BACKSPACE:
#                    user_input = ""

#                elif event.key == pygame.K_SPACE and user_input != "":
#                    return int(user_input)
