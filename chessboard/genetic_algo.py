import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .utils import get_optimal_configuration
from .plot import plot_chromosome, plot_history


class Individual(object):
    """
    A class representation of an individual
    """
    def __init__(self, polygons, board_size, chromosome=None):
        """
        Initialization function. A random chromosome is generated if no chromosome is specified.
        :param polygons: initial polygon list
        :param board_size: size off the chessboard
        :param chromosome: order sequence of the pieces
        """
        self.n_pieces = len(polygons)
        self.chromosome = self.generate_chromosome() if chromosome is None else np.array(chromosome)
        self.polygons = polygons
        self.board_size = board_size
        self.placements, self.fitness = get_optimal_configuration(self.chromosome, polygons, board_size)

    def generate_chromosome(self):
        """
        Create a random chromosome sequence
        """
        return np.random.choice(range(self.n_pieces), self.n_pieces, replace=False).astype(int)

    def mutate(self, mutation_probability):
        """
        Mutate the chromosome by swapping the gene with a random gene if a condition is met
        :param mutation_probability: probability for the mutation to occur
        """
        chromosome = self.chromosome.copy()
        has_mutated = False
        for idx, c in enumerate(chromosome):
            if np.random.rand() < mutation_probability:
                has_mutated = True
                rand_idx = np.random.choice(range(len(chromosome)))
                chromosome[idx], chromosome[rand_idx] = chromosome[rand_idx], chromosome[idx]

        if has_mutated:
            self.chromosome = chromosome
            self.placements, self.fitness = get_optimal_configuration(self.chromosome, self.polygons, self.board_size)

    def mate(self, partner):
        """
        Create offspring with a order-crossover method
        :param partner: a second p1 which will participate in the mating
        :return: two new offspring individuals
        """
        c1, c2 = np.zeros((2, self.n_pieces), dtype=int)
        start, end = np.sort(np.random.choice(range(self.n_pieces), 2, replace=False))

        p1 = self.chromosome
        p2 = partner.chromosome

        c1[start:end + 1] = p1[start:end + 1]
        mask1 = ~np.in1d(p1, p1[start:end + 1])
        mask2 = ~np.in1d(p2, p1[start:end + 1])
        c1[mask1] = p2[mask2]

        c2[start:end + 1] = p2[start:end + 1]
        mask1 = ~np.in1d(p2, p2[start:end + 1])
        mask2 = ~np.in1d(p1, p2[start:end + 1])
        c2[mask1] = p1[mask2]

        return Individual(self.polygons, self.board_size, c1), Individual(self.polygons, self.board_size, c2)

    def plot(self, filename=''):
        """
        Plot the puzzle
        :param filename: filename for saving the image
        """
        if filename == '':
            plot_chromosome(self, self.polygons, self.board_size)
        else:
            plot_chromosome(self, self.polygons, self.board_size, filename)


class Evolution(object):
    """
    A class representation of the evolution procedure
    """
    def __init__(self, n_population, polygons, board_size, mutation_probability=0.01):
        """
        Initialization function. A random chromosome is generated if no chromosome is specified.
        :param n_population: size of the population to work with
        :param polygons: initial polygon list
        :param board_size: size off the chessboard
        :param mutation_probability: probability for mutations
        """
        self.n_population = n_population + 1 if n_population % 2 != 0 else n_population
        self.polygons = polygons
        self.board_size = board_size
        self.mutation_probability = mutation_probability
        self.population = None
        self.best = None
        self.history = []

    def initialize_population(self):
        """
        Initialize a random population
        :return: a list of individuals
        """
        print(f'Generation: {len(self.history)}: Initializing')
        self.population = [Individual(self.polygons, self.board_size) for _ in tqdm(range(self.n_population),
                                                                                    leave=False)]
        self.mutate_generation()

    def select_best_pair(self):
        """
        Select two individuals from the population randomly, where the probability of choosing an individual
        is weighted by it's fitness level.
        :return: two selected individuals
        """
        pop_fitness_list = np.array([ind.fitness for ind in self.population])
        pop_fitness_list = -pop_fitness_list + np.min(pop_fitness_list) + np.max(pop_fitness_list) + 1
        pop_fitness_list = pop_fitness_list / np.sum(pop_fitness_list)
        p1, p2 = np.random.choice(self.population,
                                  size=2,
                                  replace=False,
                                  p=pop_fitness_list)
        return p1, p2

    def mutate_generation(self):
        """
        Apply a round of mutation over the whole population
        """
        for ind in self.population:
            ind.mutate(self.mutation_probability)

    def next_generation(self):
        """
        Create a next generation of offspring
        """
        print(f'Generation: {len(self.history)}')
        new_population = []
        for _ in tqdm(range(int(self.n_population/2)), leave=False):
            p1, p2 = self.select_best_pair()
            c1, c2 = p1.mate(p2)
            new_population.extend([c1, c2])
        self.population = new_population
        self.mutate_generation()

    def check_condition(self):
        """
        Check if the solution has been found
        :return: boolean
        """
        pop_fitness_list = np.array([ind.fitness for ind in self.population], dtype=int)
        if np.any(pop_fitness_list == 0):
            return True

    def get_best_candidate(self):
        """
        Get best candidate from the current population
        :return: best individual
        """
        best_id = np.argsort([ind.fitness for ind in self.population])[0]
        return self.population[best_id]

    def plot_process(self, filename=''):
        """
        Plot the evolution process
        :param filename: filename for saving the image
        """
        if filename == '':
            plot_history(np.array(self.history))
        else:
            plot_history(np.array(self.history), filename)

    def run(self, filename=''):
        """
        Run the evolution process
        :param filename: filename for saving the image
        """
        self.initialize_population()
        self.best = self.get_best_candidate()
        pop_fitness_list = np.array([ind.fitness for ind in self.population], dtype=int)
        self.history.append(pop_fitness_list)
        self.plot_process(filename)
        plt.close()
        print(f'Best candidate score: {self.best.fitness}, '
              f'        chromosome: [{",".join(np.array(self.best.chromosome, dtype=str))}]')
        while not self.check_condition():
            self.next_generation()
            best = self.get_best_candidate()
            pop_fitness_list = np.array([ind.fitness for ind in self.population], dtype=int)
            self.history.append(pop_fitness_list)
            self.plot_process(filename)
            plt.close()
            if best.fitness < self.best.fitness:
                self.best = best
            print(f'Best candidate score: {self.best.fitness}, '
                  f'chromosome: [{",".join(np.array(self.best.chromosome, dtype=str))}]')
