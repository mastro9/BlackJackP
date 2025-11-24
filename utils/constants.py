import pygame

# DIMENSIONI SCHERMATA
SCREEN_WIDTH = 1250
SCREEN_HEIGHT = 750
HALF_WIDTH = SCREEN_WIDTH / 2
HALF_HEIGHT = SCREEN_HEIGHT / 2

# COLORI

BLACK  = (0, 0, 0)
BLUE   = (51, 235, 255)
WHITE  = (255, 255, 255)
ORANGE = (255, 165, 0)
RED    = (255, 0, 0)


# FONT

pygame.font.init()
FONT_TYPE = 'Comic Sans MS'

FONT_TITLE      = pygame.font.SysFont(FONT_TYPE, 80)
FONT_HEADING    = pygame.font.SysFont(FONT_TYPE, 60)
FONT_SUBTITLE   = pygame.font.SysFont(FONT_TYPE, 45)
FONT_BOLD       = pygame.font.SysFont(FONT_TYPE, 30)
FONT_NORMAL     = pygame.font.SysFont(FONT_TYPE, 20)
FONT_SMALL      = pygame.font.SysFont(FONT_TYPE, 10)

# BACKGROUND DEL TAVOLO

POKER_BACKGROUND = pygame.transform.scale(
    pygame.image.load("Resources/Icons/pokerBackground1.jpg"),
    (SCREEN_WIDTH, SCREEN_HEIGHT)
)
POKER_BACKGROUND_GAME = pygame.transform.scale(
    pygame.image.load("Resources/Icons/pokerBackground3.jpg"),
    (SCREEN_WIDTH, SCREEN_HEIGHT)
)