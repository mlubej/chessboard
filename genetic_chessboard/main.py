import numpy as np
import matplotlib.pyplot as plt
import itertools
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint
from shapely.affinity import translate, rotate, scale
import geopandas as gpd

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


def create_polygons(fblocks):
    polygons = []
    for block in fblocks:
        pts = np.array(block)[:,:-1]
        poly = MultiPoint(pts+0.5).buffer(0.5,cap_style=3)
        poly.checkers = np.array(block)
        polygons.append(poly)
    return np.array(polygons)


board_size = 8
polygons = create_polygons(fblocks)
canvas = Polygon([[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]])
pt_grid = np.array([Point([x + 0.5, y + 0.5]) for x in range(board_size) for y in range(board_size)], dtype=object)
pos_grid = np.array([[x, y] for x in range(board_size) for y in range(board_size)], dtype=int)
import sys
import genlib
import time

start = time.time()
evo = genlib.Evolution(int(sys.argv[1]), npcs=13, mut_threshold=0.01, instance=sys.argv[2], fdir='./')
evo.darwinize()
duration = time.time() - start
print(f'Process took {duration} s.')
print(f'Best chromosome: {evo.get_best_candidate().chromosome}')
