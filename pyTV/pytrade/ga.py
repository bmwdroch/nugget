import random as r
import os

import numpy as np


class GA:
    assimilation_probability = 1
    mutation_probability = 95
    max_population_size = 200
    population_size = 100
    iterations = 50000

    overwriting = True

    def __init__(self, exchange, strategy):
        self.exchange = exchange
        self.strategy = strategy
        self.file_txt = os.path.abspath('strategies/' + strategy['name']
            + '/' + 'parameters_' + self.exchange.symbol + '_'
            + str(self.exchange.interval) + '.txt')
        self.file_py = os.path.abspath('strategies/' + strategy['name']
            + '/' + strategy['name'] + '.py')
        self.samples = [
            [r.choice(j) for j in strategy['class'].PARAMETER_VALUES.values()]
            for i in range(GA.population_size)
        ]
        self.population = {
            k: v for k, v in zip(
                map(self.fit, self.samples), self.samples
            )
        }
        self.sample_length = len(strategy['class'].PARAMETER_VALUES)
        self.actual_population_size = len(self.population)
        self.best_score = float('-inf')

    def fit(self, sample):
        strategy = self.strategy['class'](self.exchange, sample)
        return strategy.net_profit

    def select(self):
        if r.randint(0, 1) == 0:
            score = max(self.population)
            parent_1 = self.population[score]
            population_copy = self.population.copy()
            del population_copy[score]
            parent_2 = r.choice(list(population_copy.values()))
            self.parents = [parent_1, parent_2]
        else:
            self.parents = r.sample(list(self.population.values()), 2)

    def cross(self):
        r_number = r.randint(0, 1)

        if r_number == 0:
            delimiter = r.randint(1, self.sample_length - 1)
            self.child = (self.parents[0][:delimiter] 
                        + self.parents[1][delimiter:])
        else:
            delimiter_1 = r.randint(1, self.sample_length // 2 - 1)
            delimiter_2 = r.randint(
                self.sample_length // 2 + 1, self.sample_length - 1)
            self.child = (self.parents[0][:delimiter_1]
                        + self.parents[1][delimiter_1:delimiter_2]
                        + self.parents[0][delimiter_2:])

    def mutate(self):
        if r.randint(0, 100) < GA.mutation_probability:
            gene_number = r.randint(0, self.sample_length - 1)
            gene_value = r.choice(
                list(
                    self.strategy['class'].PARAMETER_VALUES.values()
                )[gene_number]
            )
            self.child[gene_number] = gene_value

    def expand(self):
        score = self.fit(self.child)
        self.population[score] = self.child

    def assimilate(self):
        if r.randint(0, 100) < GA.assimilation_probability:
            samples = [
                [
                    r.choice(j) 
                        for j in self.strategy['class'].PARAMETER_VALUES.values()
                ]
                for i in range(len(self.population) // 2)
            ]
            population = {k: v for k, v in zip(map(self.fit, samples), samples)}
            self.population.update(population)

    def elect(self):
        if self.best_score < max(self.population):
            self.best_score = max(self.population)
            print(
                'Iteration #' + str(self.iteration) +
                ', Net profit, $: ' + str(self.best_score)
            )

    def kill(self):
        while len(self.population) > GA.max_population_size:
            del self.population[min(self.population)]

    def write(self):
        self.best_sample = self.population[self.best_score]
        file_text = ('Period: ' + self.exchange.start + ' — ' +
            self.exchange.end + '\n' + 'Net profit, $: ' +
            str(self.best_score) + '\n' + 'parameter values:' + '\n')
        file_text += ''.join(
            [
                value + ' — ' + str(self.best_sample[count]) + '\n'
                    for count, value in enumerate(
                        self.strategy['class'].PARAMETER_VALUES.keys()
                    )
            ]
        )
        file_text += ''.join('=' * 50)

        with open(self.file_txt, 'a') as f:
            print(file_text, file=f)

    def overwrite(self):
        for count, value in enumerate(
            self.strategy['class'].PARAMETER_VALUES.keys()
        ):
            with open(self.file_py, 'r') as f:
                content = f.read().rstrip('\n')

            with open(self.file_py, 'w') as f:
                parameter = self.best_sample[count]

                if not isinstance(parameter, np.ndarray):
                    key = value + ' = '
                    index = content.find(key)
                    old = content[index:content.find('\n', index)]
                    new = key + str(parameter)
                    print(content.replace(old, new), end='', file=f)
                else:
                    key = value + ' = np.array('
                    index = content.find(key)
                    old = content[index:content.find('\n', index)]
                    new = key + str(list(parameter)) + ')'
                    print(content.replace(old, new), end='', file=f)

    def start(self):
        for i in range(GA.iterations):
            self.iteration = i + 1
            self.select()
            self.cross()
            self.mutate()
            self.expand()
            self.assimilate()
            self.elect()
            self.kill()
        
        self.write()

        if GA.overwriting:
            self.overwrite()