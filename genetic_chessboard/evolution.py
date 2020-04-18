import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from .individual import Individual
from .plot import plot_history


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
        self.generation = 0
        self.history = []

    def initialize_population(self):
        """
        Initialize a random population
        :return: a list of individuals
        """

        self.population = [Individual(self.polygons, self.board_size) for i in tqdm(range(self.n_population),
                                                                                    leave=False)]

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
        new_population = []
        for i in tqdm(range(int(self.n_population/2)), leave=False):
            p1, p2 = self.select_best_pair()
            c1, c2 = p1.mate(p2)
            new_population.extend([c1, c2])
        self.population = new_population
        self.mutate_generation()
        self.generation += 1

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
        if filename == '':
            plot_history(np.array(self.history))
        else:
            plot_history(np.array(self.history), filename)

    def run(self, filename=''):
        print('Initializing')
        self.initialize_population()
        self.mutate_generation()
        self.best = self.get_best_candidate()
        pop_fitness_list = np.array([ind.fitness for ind in self.population], dtype=int)
        self.history.append([self.generation, pop_fitness_list])
        self.plot_process(filename)
        plt.close()
        print(f'Generation: {self.generation}, best candidate score: {self.best.fitness}, '
              f'        chromosome: [{",".join(np.array(self.best.chromosome, dtype=str))}]')
        while not self.check_condition():
            print('Creating next generation')
            self.next_generation()
            best = self.get_best_candidate()
            pop_fitness_list = np.array([ind.fitness for ind in self.population], dtype=int)
            self.history.append([self.generation, pop_fitness_list])
            self.plot_process(filename)
            plt.close()
            if best.fitness < self.best.fitness:
                self.best = best
                print(f'Generation: {self.generation}, best candidate score: {self.best.fitness}, '
                      f'chromosome: [{",".join(np.array(self.best.chromosome, dtype=str))}]')
