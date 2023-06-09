#!/usr/bin/env python3
'''
An example of Genetic Algorithms from scratch using Python3
'''

import random
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
NUM_GENERATIONS = 100


class City:
    '''
    City class to retrieve name and 2D coordinates
    '''

    def __init__(self, name, x_pos, y_pos):
        self.name = name
        self.x_pos = x_pos
        self.y_pos = y_pos

    def distance_to(self, city):
        '''
        Compute distance between current city
        and reference city
        '''
        x_diff = abs(self.x_pos - city.x_pos)
        y_diff = abs(self.y_pos - city.y_pos)
        return (x_diff ** 2 + y_diff ** 2) ** 0.5

    def plot_city(self):
        '''
        Method for plot current city
        '''

        plt.scatter(self.x_pos, self.y_pos, label=self.name)


class GeneticAlgorithm:
    '''
    Genetic Algorithm class
    '''

    def __init__(self, cities):
        self.cities = cities
        self.population = []

    def initial_population(self):
        '''
        Generate initial population method
        '''

        for _ in range(POPULATION_SIZE):
            individual = random.sample(self.cities, len(self.cities))
            self.population.append(individual)

    def fitness_function(self, individual):
        '''
        Calculate fitness
        '''

        total_distance = 0
        for i, element in enumerate(individual):
            current_city = element
            next_city = individual[(i + 1) % len(individual)]
            distance = current_city.distance_to(next_city)
            total_distance += distance
        return 1 / total_distance

    def selection(self):
        '''
        Selection method
        '''

        logger.info('Starting selection phase')
        sample = random.choices(self.population,
                                  weights=[self.fitness_function(individual)
                                           for individual in self.population],
                                  k=2)
        self.print_list(text='Parent1: ',
                        parents=sample[0])
        self.print_list(text='Parent2: ',
                        parents=sample[1])
        return sample[0], sample[1]

    def print_list(self,
                   parents: list,
                   text: str = ''):
        '''
        Print the name of cities into the list
        '''

        try:
            names = [x.name for x in parents]
            text += ','.join(names)
            logger.info(text)
        except AttributeError:
            logger.warning(parents)


    def crossover(self, parent1, parent2):
        '''
        Crossover method
        '''

        logger.debug('Starting crossover phase')
        child = [None] * len(parent1)
        start_pos = random.randint(0, len(parent1) - 1)
        end_pos = random.randint(start_pos + 1, len(parent1))

        child[start_pos:end_pos] = parent1[start_pos:end_pos]
        for parent2_element in parent2:
            if parent2_element not in child:
                for j, _ in enumerate(child):
                    if child[j] is None:
                        child[j] = parent2_element
                        break

        self.print_list(text='Generated child: ',
                        parents=child)
        return child

    def mutate(self, individual):
        '''
        Mutation method
        '''

        for element in individual:
            if random.random() < MUTATION_RATE:
                j = random.randint(0, len(individual) - 1)
                element, individual[j] = individual[j], element

    def evolve(self):
        '''
        Evolve method
        '''

        self.initial_population()

        for generation in range(NUM_GENERATIONS):
            logger.info('Starting generation %s', generation)
            new_population = []

            for _ in range(len(self.population)):
                parent1, parent2 = self.selection()
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)

            self.population = new_population

        best_individual = max(self.population, key=self.fitness_function)
        return best_individual


def plot_best_route(best_route: list,
                    distance: int,
                    figurename: str = 'solution.png'):
    '''
    Plot cities coordinates of the best route
    '''

    plt.figure()
    for city in best_route:
        city.plot_city()

    plt.legend()
    plt.title(f'Distance = {distance}')
    plt.savefig(figurename)


def compute_total_distance(best_route: list) -> float:
    '''
    Compute the total distance of best route
    '''

    distance = 0
    for idx in range(len(best_route)-1):
        distance += best_route[idx].distance_to(best_route[idx + 1])

    return distance


def main():
    '''
    Main function
    '''

    cities = [
        City('A', 1, 1),
        City('B', 1, 2),
        City('C', 2, 3),
    ]

    genetic_algorithm = GeneticAlgorithm(cities)
    best_route = genetic_algorithm.evolve()

    distance = compute_total_distance(best_route)
    plot_best_route(best_route, distance)

    names = [city.name for city in best_route]
    logger.info("Best route: %s", names)
    logger.info("Distance = %s", distance)


if __name__ == '__main__':
    main()
