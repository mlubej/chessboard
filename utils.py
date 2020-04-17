import itertools
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint
from shapely.affinity import translate, rotate, scale


def create_polygons(center_blocks):
    """
    Create shapely Polygons from center points, where center points
    consist of a collection of tuplex (x, y, m), where x and y are the
    coordinates on the board, and m is a checker value (0/1)
    :param center_blocks: collection of center points with (x, y, m)
    :return: np.array of initial_polygons
    """
    polygons = []
    for block in center_blocks:
        pts = np.array(block)[:, :-1]
        poly = MultiPoint(pts+0.5).buffer(0.5, cap_style=3)
        poly.checkers = np.array(block)
        polygons.append(poly)
    return np.array(polygons)


def transform(poly, vec, r, f):
    """
    Apply a transformation to the polygon
    :param poly: Polygon to be transformed
    :param vec: translation vector
    :param r: angle of rotation (in degrees)
    :param f: left-right mirror flip (-1/1)
    :return: transformed polygon
    """
    poly = rotate(poly, r, origin=np.array([0.5, 0.5]))
    poly = scale(poly, f, origin=np.array([0.5, 0.5, 0.0]))
    poly = translate(poly, *vec)
    return poly


def calculate_outline(profile):
    """
    Calculate the exterior and interior outlines of the profile
    :param profile: shapely Polygon
    :return: boundary length
    """
    outline = profile.exterior.length
    for interior in profile.interiors:
        outline += interior.length
    return outline


def get_outline_and_new_profile(profile, poly, setting):
    """
    Function returns the outline score of a particular polygon with a particular setting
    placed on a particlular profile. If the poly doesn't fit inside the profile or if
    the poly splits the profile into multiple pieces, a tuple of (np.nan, None) is returned,
    otherwise the tuple of the outline and the new profile.
    :param profile: initial profile
    :param poly: input polygon
    :param setting: transformation to be applied ((x, y), r, f)
    :return: tuple of new profile length and new profile shape
    """
    poly = transform(poly, *setting)
    if not poly.within(profile):
        return np.nan, None

    new_profile = profile.difference(poly)
    if type(new_profile) == MultiPolygon:
        return np.nan, None

    outline = calculate_outline(new_profile)
    return outline, new_profile


def get_unique_orientations(poly):
    """
    Out of all possible rotations and flips, take into account the polygon symmetry and
    obtain the unique polygon orientations.
    :param poly: input polygon
    :return: array of unique settings of (rotation, flip)
    """
    rots = [0, 90, 180, 270]
    flips = [1, -1]
    iterables = [rots, flips]
    settings = np.array(list(itertools.product(*iterables)))
    unique_polys = []
    unique_settings = []
    for rot, flip in settings:
        p = poly
        p = rotate(p, rot)
        p = scale(p, flip)
        if np.any([p.equals(u) for u in unique_polys]):
            continue
        unique_polys.append(p)
        unique_settings.append([rot, flip])
    return unique_settings


def optimal_placement(profile, poly, board_size, origin_checker=None):
    """
    Function for finding the optimal placement of a polygon in the profile. If polygon cannot be placed
    in the given profile, return None, otherwise return the setting of the optimal placement and
    the corresponding profile.
    :param profile: input profile
    :param poly: input polygon
    :param board_size: size of initial board
    :param origin_checker: bottom left checker value (default: None)
    :return: the setting of the optimal placement and the corresponding profile
    """
    point_grid = np.array([Point([x + 0.5, y + 0.5]) for x in range(board_size) for y in range(board_size)],
                          dtype=object)
    position_grid = np.array([[x, y] for x in range(board_size) for y in range(board_size)], dtype=int)
    point_mask = np.array([pt.within(profile) for pt in point_grid], dtype=bool)

    if origin_checker is not None:
        if origin_checker == poly.checkers[0,-1]:
            point_mask &= np.sum(position_grid, axis=-1) % 2 == 0
        else:
            point_mask &= np.sum(position_grid, axis=-1) % 2 == 1

    rflips = get_unique_orientations(poly)
    pos = position_grid[point_mask]

    iterables = [pos, rflips]
    settings = np.array(list(itertools.product(*iterables)))
    settings = np.array([[s[0], *s[1]] for s in settings])

    scores, profiles = np.array([get_outline_and_new_profile(profile, poly, s) for s in settings]).T
    scores = scores.astype(float)
    opt_id = np.argsort(scores)[0]
    if profiles[opt_id] is not None:
        return settings[opt_id], profiles[opt_id]


def cantor(a, b):
    """
    Cantor pairing function for getting unique numbers from a pair of input numbers.
    :param a: input number
    :param b: input number
    :return: output number
    """
    return 0.5 * (a + b + 1) * (a + b) + b


def get_optimal_configuration_and_fitness(chromosome, initial_polygons, board_size):
    """
    Put each piece in the chromosome order in the optimal place until the final configuration is
    achieved. return the fitness level and the corresponding placements.
    :param chromosome:
    :param initial_polygons:
    :param board_size:
    :return:
    """
    profile = Polygon([[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]])
    placements = np.full_like(chromosome, None, dtype=object)
    origin_checker = None
    for idx, c in enumerate(chromosome):
        p = initial_polygons[c]
        try:
            opt, profile = optimal_placement(profile, p, board_size, origin_checker)
            placements[idx] = opt
            if origin_checker is None:
                sum_vec = np.sum(opt[0])
                if sum_vec % 2 == p.checkers[0, -1]:
                    origin_checker = 0
                else:
                    origin_checker = 1
        except:
            break
    empty_area = profile.area
    unused_pieces = len(chromosome) - idx - 1
    return cantor(empty_area, unused_pieces), placements
