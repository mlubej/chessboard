import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Polygon

from .utils import transform


def plot_chromosome(individual, initial_polygons, board_size, filename=''):
    """
    Function to plot the final configuration for a given chromosome.
    :param individual: an individual class, containing information about the pieces order and placement
    :param initial_polygons: original definition of polygons
    :param board_size: size of chessboard
    :param filename: name of the file to save the image to
    """
    new_polys = []
    for poly in initial_polygons:
        pts = np.array(poly.checkers)[:, :-1]
        checkers = np.array(poly.checkers)[:, -1]
        pts = pts[checkers == 1]
        poly = poly.difference(MultiPoint(pts + 0.5).buffer(0.25))
        new_polys.append(poly)

    new_polys = np.array(new_polys)
    new_polys = [transform(p, *s) for p, s in zip(new_polys[individual.chromosome],
                                                  individual.placements) if s is not None]
    indices = [individual.chromosome[idx] for idx, s in enumerate(individual.placements) if s is not None]
    canvas = Polygon([[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]])
    gdf = gpd.GeoDataFrame({'idx': indices}, geometry=new_polys)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(*canvas.exterior.xy, 'k')
    gdf.plot(ax=ax, column='idx', edgecolor='black', alpha=0.75, vmin=0, vmax=len(initial_polygons), cmap=plt.cm.hot)
    ax.set_ylim([-0.1, board_size + 0.1])
    ax.set_xlim([-0.1, board_size + 0.1])
    ax.set_title(f'[{",".join(individual.chromosome.astype(str))}]', fontsize=20)
    ax.set_xlabel(f'Fitness score: {int(individual.fitness)}', fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])

    if filename != '':
        fig.savefig(filename, dpi=300, bbox_inches='tight')


def plot_history(history, filename=''):
    """
    Function to plot the history of the evolution process
    :param history: input data for each generation
    :param filename: filename for saving the image
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for gen in history:
        ax.scatter(np.full_like(gen[1], gen[0]), gen[1], s=50, c='k', marker='o')
        ax.set_title('Evolution process', fontsize=20)
        ax.set_xlabel('Generation number', fontsize=20)
        ax.set_ylabel('Fitness level', fontsize=20)
        ax.set_ylim(bottom=0.0)
    if filename != '':
        fig.savefig(filename)
