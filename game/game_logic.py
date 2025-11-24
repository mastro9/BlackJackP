import pygame
from utils.constants import *
from utils.helpers import draw_text



# function to draw the screen every time an action is conducted in the playing of the game
def draw_table(screen, dealer, players):
    screen.blit(POKER_BACKGROUND_GAME, (0, 0))

    dealer.drawHand(screen)
    for p in players:
        p.drawHand(screen)
    # --- TIGER EYE (come nel gioco originale) ---
    tiger = pygame.image.load("Resources/Icons/tigerEye.png")

    # riduzione a metà della dimensione
    tiger_w = round(tiger.get_width() * 0.5)
    tiger_h = round(tiger.get_height() * 0.5)

    tiger = pygame.transform.scale(tiger, (tiger_w, tiger_h))

    # trasparenza per l'effetto "velato"
    tiger.set_alpha(60)

    # centrato sul tavolo
    screen.blit(tiger,(HALF_WIDTH - tiger_w / 2, HALF_HEIGHT - tiger_h / 2))
    pygame.display.update()


# function to create the hands of the dealer and all the players
def deal_initial_cards(players, dealer):
    dealer.createDealerHand()
    for _ in range(2):
        for p in players:
            p.addCard(dealer.dealCard())


def check_blackjack(players):
    for p in players:
        if p.count == 21:
            p.blackjack = True
            p.applyBet(3/2)
            p.resetBet()


def play_turns(players, dealer):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    turn = 0
    while turn < len(players):
        player = players[turn]

        if player.blackjack:
            turn += 1
            continue

        # evidenzia chi sta giocando
        for p in players:
            p.currentTurn = (p == player)

        draw_table(screen, dealer, players)

        #richiesta se hit o pass
        choice = player.askChoice()

        # HIT
        if choice == 1:
            while True:
                card = dealer.dealCard()
                player.addCard(card)
                draw_table(screen, dealer, players)

                if player.count > 21:
                    player.bust = True
                    player.resetBet()
                    break

                choice = player.askChoice()
                if choice != 1:
                    break

        turn += 1


def resolve_round(players, dealer):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    # --- FASE 1: Dealer scopre la carta coperta ---
    dealer.reveal_all_cards()
    draw_table(screen, dealer, players)
    pygame.time.delay(700)  # piccolo effetto

    # --- CONTROLLO: ci sono giocatori ancora "vivi"? ---
    # giocatori non bust e con una bet ancora attiva
    active_players = [p for p in players if not p.bust]

    # se tutti hanno sballato (o non hanno più puntata), il banco vince di default
    if not active_players:
        
        draw_table(screen, dealer, players)
        pygame.time.delay(3000)

        return

    # --- FASE 2: Dealer pesca fino a 17 ---
    while dealer.count <= 16:
        dealer.addCard()

        # mostra aggiornamento
        draw_table(screen, dealer, players)
        pygame.time.delay(3000)  # tempo per "vedere" la carta

    # --- FASE 3: Dealer bust? ---
    if dealer.count > 21:
        for p in players:
            if p.bet > 0:
                p.applyBet(2)
                p.resetBet()
        return

    # --- FASE 4: confronto punteggi ---
    highest = max((p.count for p in players if not p.bust and not p.blackjack), default=0)

    for p in players:

        # blackjack già pagato
        if p.blackjack or p.bust:
            continue

        if p.count == highest and highest > dealer.count:
            p.applyBet(2)
            p.resetBet()
        elif p.count == dealer.count:
            p.applyBet(1)
            p.resetBet()
        else:
            p.resetBet()