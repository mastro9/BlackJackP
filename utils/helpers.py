import pygame
from utils.constants import *
import sys


def draw_text(surface, text, font, color, x, y):
    """Disegna testo centrato sulla coordinata (x,y)."""
    textObject = font.render(text, False, color)
    textWidth = textObject.get_rect().width
    textHeight = textObject.get_rect().height
    surface.blit(textObject, (x - (textWidth / 2), y - (textHeight / 2)))



def load_image(path, scale=None):
    """Carica immagine, opzionalmente la scala."""
    img = pygame.image.load(path)
    if scale:
        img = pygame.transform.scale(img, scale)
    return img



def wait_for_input():
    """Aspetta SPACE per continuare oppure ESC/chiudi finestra per uscire."""
    waiting = True
    while waiting:
        for event in pygame.event.get():

            # chiusura finestra
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # tasti
            if event.type == pygame.KEYDOWN:

                # ESC = esci
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                # SPACE = continua
                if event.key == pygame.K_SPACE:
                    waiting = False