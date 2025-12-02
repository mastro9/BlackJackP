import random

class BlackjackEnv:
    def __init__(self):
        # Valori delle carte: J, Q, K valgono 10. Asso vale 11 (gestito dopo)
        self.deck_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    def _pesca_carta(self):
        return random.choice(self.deck_values)

    def _calcola_punteggio(self, mano):
        # Logica per gestire l'Asso
        punteggio = sum(mano)
        # Se ho un Asso e sballo (supero 21), l'Asso diventa 1
        if 1 in mano and punteggio <= 11:
            # Qui stiamo assumendo che l'asso nella lista sia 1. 
            # Nel blackjack standard spesso si conta 11 se non sballi.
            # Per semplificare all'AI: se c'è un 1 e aggiungendo 10 non sballo, usalo come 11.
            return punteggio + 10
        return punteggio

    def _ha_asso_usabile(self, mano):
        # True se ho un asso che sta contando come 11
        punteggio = sum(mano)
        return 1 in mano and punteggio + 10 <= 21

    def reset(self):
        """
        Inizia una nuova partita.
        Restituisce: la tupla di stato iniziale.
        """
        self.player_hand = [self._pesca_carta(), self._pesca_carta()]
        self.dealer_hand = [self._pesca_carta(), self._pesca_carta()]
        
        return self._get_obs()

    def _get_obs(self):
        """
        Restituisce la tupla (Stato) che vede l'AI.
        """
        punteggio_player = self._calcola_punteggio(self.player_hand)
        dealer_card = self.dealer_hand[0] # L'AI vede solo la prima carta
        asso_usabile = self._ha_asso_usabile(self.player_hand)
        
        # Esempio Tupla: (14, 10, False)
        return (punteggio_player, dealer_card, asso_usabile)

    def step(self, azione):
        """
        Esegue un'azione nel gioco.
        Azione: 0 = Stai (Stand), 1 = Carta (Hit)
        Restituisce: (nuovo_stato, ricompensa, fine_partita)
        """
        # AZIONE 1: HIT (CARTA)
        if azione == 1:
            self.player_hand.append(self._pesca_carta())
            punteggio = self._calcola_punteggio(self.player_hand)
            
            if punteggio > 21:
                # SBALLATO: Fine gioco, Ricompensa negativa
                return self._get_obs(), -1, True
            else:
                # GIOCO CONTINUA: Ricompensa 0 (per ora), Fine False
                return self._get_obs(), 0, False

        # AZIONE 0: STAND (STAI)
        else:
            # Tocca al Dealer
            punteggio_player = self._calcola_punteggio(self.player_hand)
            punteggio_dealer = self._calcola_punteggio(self.dealer_hand)
            
            # Il dealer tira finché ha meno di 17
            while punteggio_dealer < 17:
                self.dealer_hand.append(self._pesca_carta())
                punteggio_dealer = self._calcola_punteggio(self.dealer_hand)
            
            # CALCOLO VINCITORE
            reward = 0
            if punteggio_dealer > 21: # Dealer sballa
                reward = 1
            elif punteggio_dealer > punteggio_player: # Dealer vince
                reward = -1
            elif punteggio_dealer < punteggio_player: # Player vince
                reward = 1
            else: # Pareggio
                reward = 0
                
            return self._get_obs(), reward, True