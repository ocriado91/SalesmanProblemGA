# SalesmanProblemGA
Travelling Salesman Problem through Genetic Algorithms

## Introduction
__Genetic Algorithms (GAs)__ are computational techniques that draw inspiration
from the principles of natural selection and genetics, as famously explored
by Charles Darwin in his theory of evolution. These algorithms provide a
powerful framework for solving complex optimization problems in diverse fields
such as engineering, finance, biology, and artificial intelligence.

The process of natural selection starts with the selection of fittest individuals
from a population. They produce offspring which inherit the chracteristics of the
parents and will be added to the next generation. If parents have better fitness,
their offspring will be better than parents and have a better of chance at surviving.
This process keeps on iterating and at the end, a generation with the fittest
individuals will be found.

## Phases of Genetic Algorithms

### Initial population
The process begins with a set of individuals which is called a **Population**.
Each individual is a solution to the problem you want to solve.

An individual is characterized by a set of parameters (variables) known as
**Genes**. Genes are joined into a string to form a **Chromosome** (solution).

In a genetic algorithm, the set of genes of an individual is represented using a
string, in terms of an alphabet. Usually, binary values are used. We say that
we encoded the genes in a chromosome.

### Fitness function
The **fitness function** determines how fit an individual is. It gives a **fitness
score** to each individual. The probability that an individual will be selected
for reproduction is based on its fitness score.

### Selection
The idea of **selection** phase is to select the fittest individuals and let them
pass their genes to the next generation.

Two pairs of individuals (**parents**) are selected based on their fitness scores.
Individuals with high fitness have more change to be selected for reproduction.

### Crossover
**Crossover** is the most significant phase in a genetic algorithm. For each pair
of parents to be mated, a **crossover point** is chosen at random from within the
genes.

**Offspring** are created by exchanging the genes of parents among themselves until
the crossover point is reached.
### Mutation
In certain new offspring formed, some of theirs genes can be subjected to a
**mutation** with a low random probability. This implies that some of the bits
in the bit string can be flipped.



