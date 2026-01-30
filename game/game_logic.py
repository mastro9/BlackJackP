import pygame
from utils.constants import *
from utils.helpers import draw_text



# function to draw the screen every time an action is conducted in the playing of the game
def draw_table(screen, dealer, players):
    screen.blit(POKER_BACKGROUND_GAME, (0, 0))

    dealer.drawHand(screen)

    # --- AI MOD: Recuperiamo la carta scoperta del dealer ---
    # La mano del dealer ha 2 carte, la prima è quella scoperta (di solito)
    dealer_up_card = dealer.hand[0] if dealer.hand else None
    for p in players:
        # Passiamo la carta del dealer alla funzione di disegno
        p.drawHand(screen, dealer_up_card)
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
        if player.is_human:
            choice = player.askChoice()
        else:
            choice = player.autoChoice(dealer.hand[0].value)
            pygame.time.wait(700)  # piccola pausa per vedere l’azione a schermo

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

                if player.is_human:
                    choice = player.askChoice()
                else:
                    choice = player.autoChoice(dealer.hand[0].value)
                    pygame.time.wait(700)
                if choice != 1:
                    break

        turn += 1


def resolve_round(players, dealer):
    screen = pygame.display.get_surface() # Recupera la finestra attiva
    
    # --- FASE 1: Dealer scopre la carta coperta ---
    dealer.reveal_all_cards()
    draw_table(screen, dealer, players)
    pygame.time.delay(1000) 

    # --- FASE 2: Dealer pesca fino a 16 ---
    while dealer.count <= 16:
        dealer.addCard() 
        draw_table(screen, dealer, players)
        pygame.time.delay(1000)

    # --- FASE 3: Calcolo Vincite e Messaggi Grafici ---
    startY = 250 # Punto di partenza per i messaggi a video
    
    # Se il dealer sballa
    if dealer.count > 21:
        draw_text(screen, "DEALER BUSTED!", FONT_SUBTITLE, RED, HALF_WIDTH, startY)
        pygame.display.update()
        pygame.time.delay(1000)
        
        for p in players:
            if not p.bust:
                p.applyBet(2)
                p.resetBet()
                # Messaggio individuale opzionale
                startY += 40
                draw_text(screen, f"{p.name} wins 2x bet!", FONT_NORMAL, WHITE, HALF_WIDTH, startY)
    
    else:
        # Confronto normale (Logica della tua compareCounts)
        highest_count = max((p.count for p in players if not p.bust and not p.blackjack), default=0)

        for p in players:
            if p.bust or p.blackjack:
                continue
            
            startY += 40
            if p.count > dealer.count:
                draw_text(screen, f"{p.name} won twice the bet!", FONT_NORMAL, GREEN, HALF_WIDTH, startY)
                p.applyBet(2)
            elif p.count == dealer.count:
                draw_text(screen, f"{p.name} got the bet back.", FONT_NORMAL, WHITE, HALF_WIDTH, startY)
                p.applyBet(1)
            else:
                draw_text(screen, f"Dealer took {p.name}'s bet.", FONT_NORMAL, RED, HALF_WIDTH, startY)
            
            p.resetBet()

    pygame.display.update()
    pygame.time.delay(3000) # Pausa per leggere i risultati