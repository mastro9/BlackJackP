import random

class BlackjackEnv:
    def __init__(self):
        # Card values: J, Q, K are worth 10. Ace is worth 11 (handled later)
        self.deck_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    def _pesca_carta(self):
        return random.choice(self.deck_values)

    def _calcola_punteggio(self, mano):
        # Logic to handle the Ace
        punteggio = sum(mano)
        # If I have an Ace and I bust (go over 21), the Ace becomes 1
        if 1 in mano and punteggio <= 11:
            # Here we are assuming the ace in the list is 1. 
            # In standard blackjack, it is often counted as 11 if you don't bust.
            # To simplify for the AI: if there is a 1 and adding 10 doesn't bust, use it as 11.
            return punteggio + 10
        return punteggio

    def _ha_asso_usabile(self, mano):
        # True if I have an ace that is counting as 11
        punteggio = sum(mano)
        return 1 in mano and punteggio + 10 <= 21

    def reset(self):
        """
        Starts a new game.
        Returns: the initial state tuple.
        """
        self.player_hand = [self._pesca_carta(), self._pesca_carta()]
        self.dealer_hand = [self._pesca_carta(), self._pesca_carta()]
        
        return self._get_obs()

    def _get_obs(self):
        """
        Returns the tuple (State) that the AI sees.
        """
        punteggio_player = self._calcola_punteggio(self.player_hand)
        dealer_card = self.dealer_hand[0] # The AI sees only the first card
        asso_usabile = self._ha_asso_usabile(self.player_hand)
        
        # Tuple Example: (14, 10, False)
        return (punteggio_player, dealer_card, asso_usabile)

    def step(self, azione):
        """
        Executes an action in the game.
        Action: 0 = Stand, 1 = Hit
        Returns: (new_state, reward, game_over)
        """
        # ACTION 1: HIT (CARD)
        if azione == 1:
            self.player_hand.append(self._pesca_carta())
            punteggio = self._calcola_punteggio(self.player_hand)
            
            if punteggio > 21:
                # BUSTED: Game over, Negative reward
                return self._get_obs(), -1, True
            else:
                # GAME CONTINUES: Reward 0 (for now), Over False
                return self._get_obs(), 0, False

        # ACTION 0: STAND
        else:
            # Dealer's turn
            punteggio_player = self._calcola_punteggio(self.player_hand)
            punteggio_dealer = self._calcola_punteggio(self.dealer_hand)
            
            # The dealer draws until they have at least 17
            while punteggio_dealer < 17:
                self.dealer_hand.append(self._pesca_carta())
                punteggio_dealer = self._calcola_punteggio(self.dealer_hand)
            
            # WINNER CALCULATION
            reward = 0
            if punteggio_dealer > 21: # Dealer busts
                reward = 1
            elif punteggio_dealer > punteggio_player: # Dealer wins
                reward = -1
            elif punteggio_dealer < punteggio_player: # Player wins
                reward = 1
            else: # Tie
                reward = 0
                
            return self._get_obs(), reward, True