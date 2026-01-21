import pygame
import random
import sys # Necessario per sys.exit() nelle funzioni in fondo

# Import non modificati
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

        # --- GIOCATORE UMANO ---
        # *** Importante: get_player_name() deve ritornare un oggetto Player, non una lista ***
        human = get_player_name() 
        self.players.append(human)
        human.is_human = True

        # --- ALTRI 3 BOT ---
        for i in range(1, 4):
            # Assumendo che Player inizi con un bankroll predefinito > 0
            bot = Player(f"Player{i}") 
            self.players.append(bot)
        
        self.fix_coordinates(num)

        while not self.game_over:
            
            # 1. CONTROLLO E RIMOZIONE DEI GIOCATORI IN BANCAROTTA
            self.remove_bankrupt_players()
            
            # Se rimane solo un giocatore (o nessuno) dopo il check, il gioco finisce.
            # Se la lista è vuota, usciamo dal ciclo.
            if len(self.players) < 1:
                print("Tutti i giocatori sono falliti. Fine del gioco.")
                self.game_over = True
                continue # Passa al prossimo ciclo (che uscirà)

            # 2. GESTIONE PUNTATE
            # La funzione get_bet gestisce la puntata del giocatore umano (self.players[0])
            # Nota: get_bet deve gestire il caso in cui il giocatore umano ha 0 soldi
            get_bet(self.players) 
            
            # Puntate dei BOT
            # Iteriamo sulla lista aggiornata, escludendo il giocatore umano (che ha già puntato)
            for p in self.players[1:]:
                # CONTROLLO SICUREZZA AGGIUNTIVO PER I BOT (dovrebbe essere sempre p.bank > 0 qui)
                if p.bank <= 0:
                    continue # salta il bot se è in bancarotta (anche se è stato rimosso)

                bet = random.randint(1, min(100, int(p.bank)))
                p.bet = bet
                p.bank -= bet

            # 3. LOGICA DEL ROUND
            # nuovo mazzo
            self.dealer = Dealer()

            deal_initial_cards(self.players, self.dealer)
            check_blackjack(self.players)
            play_turns(self.players, self.dealer)

            resolve_round(self.players, self.dealer)
            
            # Passiamo un flag per mostrare il messaggio di uscita corretto se il gioco è finito
            show_end_round(self.players, self.dealer, game_over=self.game_over or len(self.players) < 1)

            # 4. PREPARAZIONE NUOVO ROUND
            self.reset_for_new_round()
            self.check_final_winner()
            
    def remove_bankrupt_players(self):
        """Rimuove i giocatori che hanno un bankroll di 0 o meno."""
        active_players = []
        bankrupt_players = []
        
        for p in self.players:
            if p.bank <= 0:
                bankrupt_players.append(p)
                print(f"💰 {p.name} è in bancarotta (${p.bank}) ed è stato rimosso dal gioco.")
            else:
                active_players.append(p)
                
        self.players = active_players
        
        # Dopo la rimozione, potremmo dover riassegnare le coordinate ai giocatori rimasti
        self.fix_coordinates(len(self.players))


    def fix_coordinates(self, num):
        # La logica di fix_coordinates è rimasta la stessa, ma ora
        # viene chiamata con 'num' = len(self.players) aggiornato.
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

        # ... (il resto della funzione fix_coordinates è invariato) ...
        # Ho lasciato il resto della funzione fix_coordinates come fornita

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

    # vinci con 200$
    def check_final_winner(self):
        for p in self.players:
            if p.bank >= 200:
                print(f"🎉 {p.name} ha vinto!")
                self.game_over = True
                
# --- FUNZIONI ESTERNE (modifiche in get_bet) ---

# ... show_start_screen, show_instructions, get_player_name (invariate) ...

def show_start_screen():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Welcome")

    # sfondo
    screen.blit(POKER_BACKGROUND, (0, 0))

    # carica il logo come nel gioco originale
    title_logo = pygame.image.load("Resources/Icons/titleBlitzEdition.png")
    logo_w = title_logo.get_width()
    logo_h = title_logo.get_height()

    # disegna logo centrato
    screen.blit(title_logo, (HALF_WIDTH - logo_w // 2, HALF_HEIGHT - logo_h // 2 - 25))

    # testo "premi spazio"
    draw_text(screen,"PRESS SPACE TO CONTINUE",FONT_SUBTITLE,WHITE,HALF_WIDTH,HALF_HEIGHT + 100)

    pygame.display.update()
    wait_for_input()

def show_instructions():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("How to Play")
    screen.blit(POKER_BACKGROUND, (0, 0))

    # TITOLI E REGOLE — STILE ORIGINALE
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
              "--> You can go bankrupt, but the game will always leave you with $1.",
              FONT_NORMAL, WHITE, HALF_WIDTH, 500)
    draw_text(screen,
              "--> Bust = Dealer takes your bet    Blackjack = Earn 1.5x your bet",
              FONT_NORMAL, WHITE, HALF_WIDTH, 550)
    draw_text(screen,
              "--> Closest to 21 = Earn 2x your bet, else the dealer wins.",
              FONT_NORMAL, WHITE, HALF_WIDTH, 600)
    draw_text(screen,
              "--> Tie with dealer (if highest under 21) = Get your bet back",
              FONT_NORMAL, WHITE, HALF_WIDTH, 650)
    draw_text(screen,
              "--> Dealer Bust = Everyone still in the game earns 2x their bets.",
              FONT_NORMAL, WHITE, HALF_WIDTH, 700)

    pygame.display.update()
    wait_for_input()

def get_player_name():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Enter Names")

    valid_chars = "abcdefghijklmnopqrstuvwxyz1234567890"


    name = ""
    name_confirmed = False

    while not name_confirmed:
        # SFONDO
        screen.blit(POKER_BACKGROUND, (0, 0))

        draw_text(screen,f"Enter your name:",FONT_BOLD,ORANGE,HALF_WIDTH,HALF_HEIGHT - 50)
        # NOME INSERITO
        draw_text(screen,name,FONT_BOLD,WHITE,HALF_WIDTH,HALF_HEIGHT)

        footer = "PRESS SPACE TO CONTINUE"

        draw_text(screen,footer,FONT_BOLD,ORANGE,HALF_WIDTH,HALF_HEIGHT + 50)
        pygame.display.update()

        # GESTIONE EVENTI
        for event in pygame.event.get():

            # CHIUSURA FINESTRA
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # ESC = EXIT
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

            # INSERIMENTO CARATTERI
            if event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key)

                # aggiungi carattere se valido e lunghezza < 9
                if key in valid_chars and len(name) < 9:
                        name += key

                # cancella tutto
                elif event.key == pygame.K_BACKSPACE:
                    name = ""

                # conferma nome con SPACE
                elif event.key == pygame.K_SPACE and len(name) > 0:
                    name_confirmed = True

    return Player(name)


def get_bet(players):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Enter Bets")

    valid_nums = "0123456789"
    
    # Prende il primo giocatore, che si presume essere l'umano.
    # Se la lista è vuota qui, si avrà un IndexError, ma è gestito nel GameController.start().
    p = players[0] 

    bet = ""
    bet_value = None
    valid_bet = True

    # *** PUNTO CRITICO MODIFICATO ***
    # Se il giocatore umano ha 0 o meno, non può scommettere.
    # Questo è in contraddizione con la regola "the game will always leave you with $1".
    # Se vuoi seguire la regola del $1 minimo, usa:
    if p.bank <= 0:
        print(f"ATTENZIONE: {p.name} è in bancarotta, ma la regola dice che gli resta $1.")
        # Se la regola del gioco è che resta sempre $1, assicurati che la logica di resolve_round lo faccia.
        # Qui potremmo decidere di non permettere la puntata, o forzare una puntata minima di 1.
        
        # Se l'intenzione è che *non* può giocare se è a 0, allora gestiamo solo il caso di uscita.
        # MA: La logica di rimozione in GameController gestisce già che p.bank > 0.
        # Quindi se arriviamo qui, p.bank dovrebbe essere > 0.
        pass # Non facciamo nulla qui, ci fidiamo della logica del GameController.

    while True:
        screen.blit(POKER_BACKGROUND, (0, 0))

        draw_text(screen, f"Enter {p.name}'s bet  ({p.name}'s Bank = ${p.bank})",
                          FONT_BOLD, ORANGE, HALF_WIDTH, HALF_HEIGHT - 50)
        draw_text(screen, bet, FONT_BOLD, WHITE, HALF_WIDTH, HALF_HEIGHT)

        footer = ("PRESS SPACE TO START GAME")
        draw_text(screen, footer, FONT_BOLD, ORANGE, HALF_WIDTH, HALF_HEIGHT + 50)

        if not valid_bet:
            # Messaggio per puntata non valida (es. troppo alta o zero)
            draw_text(screen, "PUNTATA NON VALIDA (Max $100 e Max Bankroll)",
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

                # numeri
                if key in valid_nums and len(bet) < 4:
                    bet += key
                    valid_bet = True

                # cancella
                elif event.key == pygame.K_BACKSPACE:
                    bet = ""
                    valid_bet = True

                # conferma
                elif event.key == pygame.K_SPACE:

                    # NON permettere puntata vuota
                    if bet == "":
                            valid_bet = False
                            continue

                    bet_value = int(bet)
                    
                    # Massima puntata è 100, e non può superare il bankroll.
                    max_allowed_bet = min(100, p.bank)

                    # minimo 1, massimo max_allowed_bet
                    if 1 <= bet_value <= max_allowed_bet:
                        p.bet = bet_value
                        p.bank -= bet_value
                        valid_bet = True
                        break
                    else:
                        # La puntata non è valida (es. 0, > 100, o > bank)
                        valid_bet = False

        if bet_value is not None and valid_bet:
            break
            
def show_end_round(players, dealer, game_over=False):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Round Over")
    screen.blit(POKER_BACKGROUND_GAME, (0, 0))

    y = 100

    # TITOLO
    draw_text(screen, "Results:", FONT_SUBTITLE, ORANGE, HALF_WIDTH, y)
    y += 60

    # DEALER
    draw_text(screen, f"Dealer's Count: {dealer.count}", FONT_NORMAL, ORANGE, HALF_WIDTH, y)
    y += 50

    num = len(players)

    # ---- RIGA 1: Max 3 giocatori ----
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

    # ---- RIGA 2: solo se ci sono più di 3 giocatori ----
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

    # ---- BANCA RIGA 1 ----
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

    # ---- BANCA RIGA 2 ----
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

    # MESSAGGIO FINALE
    footer = "PRESS SPACE TO EXIT" if game_over or num < 1 else "PRESS SPACE TO CONTINUE"
    draw_text(screen, footer, FONT_SUBTITLE, ORANGE, HALF_WIDTH, SCREEN_HEIGHT - 50)

    pygame.display.update()

    # ATTESA INPUT
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
