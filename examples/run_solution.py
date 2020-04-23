import numpy as np
import matplotlib

from chessboard.genetic_algo import Evolution, Individual
from chessboard.utils import *

matplotlib.use('Agg')

center_blocks = np.array([
    [(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1), (4, 0, 0)],
    [(0, 0, 0), (1, 0, 1), (2, 0, 0), (2, -1, 1), (1, 1, 0)],
    [(0, 0, 0), (1, 0, 1), (2, 0, 0), (0, 1, 1), (2, -1, 1)],
    [(0, 0, 0), (1, 0, 1), (2, 0, 0), (-1, 0, 1), (0, -1, 1)],
    [(0, 0, 0), (-1, 0, 1), (1, 0, 1), (0, 1, 1), (0, -1, 1)],
    [(0, 0, 1), (1, 0, 0), (2, 0, 1), (0, 1, 0), (0, -1, 0)],
    [(0, 0, 0), (1, 0, 1), (2, 0, 0), (3, 0, 1), (0, 1, 1)],
    [(0, 0, 0), (0, 1, 1), (2, 0, 0), (2, 1, 1), (1, 1, 0)],
    [(0, 0, 1), (1, 0, 0), (2, 0, 1), (0, 1, 0), (0, 2, 1)],
    [(0, 0, 1), (-1, 0, 0), (0, 1, 0), (1, 1, 1), (2, 1, 0)],
    [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0), (-1, 1, 0)],
    [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 1)],
    [(0, 0, 0), (1, 1, 0), (2, 2, 0), (0, 1, 1), (1, 2, 1)]
])

board_size = 8
polygons = create_polygons(center_blocks)

evo = Evolution(50, polygons, board_size, mutation_probability=0.01)

evo.run('./process.png')
evo.best.plot('./solution.png')
