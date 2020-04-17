import numpy as np
import matplotlib.pyplot as plt
import itertools
from shapely.geometry import Polygon, MultiPolygon, Point, MultiPoint
from shapely.affinity import translate, rotate, scale
import geopandas as gpd


class Evolution(object):
    def __init__(self, npop, npcs=len(polygons), mut_threshold=0.01, instance=0, fdir='./'):
        self.npop = npop + 1 if npop % 2 != 0 else npop
        self.npcs = npcs
        self.mut_threshold = mut_threshold
        self.population = None
        self.best = None
        self.generation = 0
        self.instance = instance
        self.history = []
        self.fdir = fdir

    def initialize_population(self):
        self.population = [Individual() for i in range(self.npop)]

    def select_best_pair(self):
        pop_fitness = np.array([ind.fitness for ind in self.population])
        pop_fitness = pop_fitness / np.sum(pop_fitness)
        p1, p2 = np.random.choice(self.population,
                                  size=2,
                                  replace=False,
                                  p=pop_fitness)
        return p1, p2

    def mutate_generation(self):
        for ind in self.population:
            ind.mutate()

    def next_generation(self):
        new_population = []
        while len(new_population) < self.npop:
            p1, p2 = self.select_best_pair()
            c1, c2 = p1.mate(p2)
            new_population.extend([c1, c2])
        self.population = new_population
        self.mutate_generation()
        self.generation += 1

    def check_condition(self):
        pop_fitness = np.array([ind.fitness for ind in self.population], dtype=int)
        if np.any(pop_fitness == 0):
            return True

    def get_best_candidate(self):
        best_id = np.argsort([ind.fitness for ind in self.population])[0]
        return self.population[best_id]

    def plot_process(self, fname=''):
        if fname == '':
            plot_process(np.array(self.history))
        else:
            plot_process(np.array(self.history), f'{self.fdir}/plot_process_{self.instance}.png')

    def darwinize(self):
        print('Initializing')
        self.initialize_population()
        self.mutate_generation()
        self.best = self.get_best_candidate()
        self.history.append([self.generation, self.best.fitness])
        self.best.plot(f'{self.fdir}/plot_sol_{self.instance}.png')
        self.plot_process(f'{self.fdir}/plot_proc_{self.instance}.png')
        print(
            f'Generation: {self.generation}, best candidate score: {self.best.fitness}, chromosome: [{",".join(np.array(self.best.chromosome, dtype=str))}]')
        while not self.check_condition():
            print('Creating next generation')
            self.next_generation()
            best = self.get_best_candidate()
            if best.fitness < self.best.fitness:
                self.best = best
                self.history.append([self.generation, self.best.fitness])
                self.best.plot(f'{self.fdir}/plot_sol_{self.instance}.png')
                self.plot_process(f'{self.fdir}/plot_proc_{self.instance}.png')
                print(
                    f'Generation: {self.generation}, best candidate score: {self.best.fitness}, chromosome: [{",".join(np.array(self.best.chromosome, dtype=str))}]')