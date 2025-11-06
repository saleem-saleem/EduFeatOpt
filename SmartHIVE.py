"""
SmartHive: ACO-GA Hybrid Feature Selection with SVM
---------------------------------------------------
Author: Saleem Malik
Description:
SmartHive is a hybrid wrapper-based optimization algorithm that
integrates Ant Colony Optimization (ACO) for global exploration
and Genetic Algorithm (GA) for local refinement to select the
most discriminative and least redundant feature subset.
An SVM classifier is used as the evaluator for feature subset fitness.
"""

import numpy as np
import random
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris  # Example dataset, replace with Student Performance
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURABLE PARAMETERS
# ============================================================
ACO_PARAMS = {
    'num_ants': 10,
    'num_iterations': 20,
    'alpha': 1.0,        # Pheromone importance
    'beta': 2.0,         # Heuristic importance
    'rho': 0.3,          # Pheromone evaporation rate
    'Q': 1.0,            # Pheromone deposit constant
}

GA_PARAMS = {
    'population_size': 10,
    'num_generations': 15,
    'crossover_rate': 0.8,
    'mutation_rate': 0.2
}

SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale'
}

# ============================================================
# STEP 1: Fitness Function
# ============================================================

def evaluate_fitness(X, y, feature_subset):
    """
    Evaluates fitness based on SVM classification accuracy and subset size.
    The fitness = weighted accuracy - penalty for feature count.
    """
    if np.sum(feature_subset) == 0:
        return 0  # No features selected → invalid

    X_selected = X[:, feature_subset == 1]
    clf = SVC(**SVM_PARAMS)
    scores = cross_val_score(clf, X_selected, y, cv=5)
    accuracy = np.mean(scores)
    
    # Penalize large feature subsets
    reduction_ratio = np.sum(feature_subset) / X.shape[1]
    fitness = 0.9 * accuracy - 0.1 * reduction_ratio
    return fitness


# ============================================================
# STEP 2: ANT COLONY OPTIMIZATION (ACO)
# ============================================================

def ant_colony_optimization(X, y, num_features):
    num_ants = ACO_PARAMS['num_ants']
    num_iterations = ACO_PARAMS['num_iterations']
    alpha = ACO_PARAMS['alpha']
    beta = ACO_PARAMS['beta']
    rho = ACO_PARAMS['rho']
    Q = ACO_PARAMS['Q']

    # Initialize pheromone levels
    pheromone = np.ones(num_features)
    heuristic = np.ones(num_features)

    best_fitness = 0
    best_solution = np.zeros(num_features)

    for iteration in range(num_iterations):
        all_solutions = []
        fitness_values = []

        for ant in range(num_ants):
            # Construct feature subset probabilistically
            probabilities = (pheromone ** alpha) * (heuristic ** beta)
            probabilities /= np.sum(probabilities)

            # Binary feature selection (stochastic sampling)
            feature_subset = (np.random.rand(num_features) < probabilities).astype(int)
            fitness = evaluate_fitness(X, y, feature_subset)

            all_solutions.append(feature_subset)
            fitness_values.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = feature_subset.copy()

        # Update pheromone trails
        pheromone = (1 - rho) * pheromone
        for i in range(num_features):
            for k in range(num_ants):
                pheromone[i] += Q * fitness_values[k] * all_solutions[k][i]

        print(f"[ACO] Iteration {iteration+1}/{num_iterations} | Best Fitness: {best_fitness:.4f}")

    return best_solution, best_fitness


# ============================================================
# STEP 3: GENETIC ALGORITHM (GA)
# ============================================================

def initialize_population(base_solution, num_features):
    population = [np.random.randint(0, 2, num_features) for _ in range(GA_PARAMS['population_size'] - 1)]
    population.append(base_solution)  # Include best ACO solution
    return population


def crossover(parent1, parent2):
    if random.random() > GA_PARAMS['crossover_rate']:
        return parent1.copy(), parent2.copy()
    point = random.randint(1, len(parent1)-2)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def mutation(chromosome):
    for i in range(len(chromosome)):
        if random.random() < GA_PARAMS['mutation_rate']:
            chromosome[i] = 1 - chromosome[i]
    return chromosome


def genetic_algorithm(X, y, initial_solution):
    num_features = X.shape[1]
    population = initialize_population(initial_solution, num_features)
    best_solution = initial_solution.copy()
    best_fitness = evaluate_fitness(X, y, initial_solution)

    for gen in range(GA_PARAMS['num_generations']):
        new_population = []

        # Evaluate fitness of population
        fitness_values = [evaluate_fitness(X, y, ind) for ind in population]

        # Select top 50% as parents
        sorted_idx = np.argsort(fitness_values)[::-1]
        parents = [population[i] for i in sorted_idx[:len(population)//2]]

        # Crossover and mutation
        while len(new_population) < GA_PARAMS['population_size']:
            p1, p2 = random.sample(parents, 2)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutation(c1))
            new_population.append(mutation(c2))

        # Keep best solution
        for ind in new_population:
            fit = evaluate_fitness(X, y, ind)
            if fit > best_fitness:
                best_fitness = fit
                best_solution = ind.copy()

        population = new_population
        print(f"[GA] Generation {gen+1}/{GA_PARAMS['num_generations']} | Best Fitness: {best_fitness:.4f}")

    return best_solution, best_fitness


# ============================================================
# STEP 4: SmartHive Main Function
# ============================================================

def SmartHive(X, y):
    num_features = X.shape[1]

    print("\n=== SMART HIVE OPTIMIZATION STARTED ===")
    print("Running ACO for global exploration...")
    best_aco_sol, aco_fit = ant_colony_optimization(X, y, num_features)

    print("\nRefining with GA for local optimization...")
    best_ga_sol, ga_fit = genetic_algorithm(X, y, best_aco_sol)

    print("\n=== SMART HIVE COMPLETED ===")
    print(f"ACO Best Fitness: {aco_fit:.4f}")
    print(f"GA Refined Fitness: {ga_fit:.4f}")
    print(f"Selected Features: {np.sum(best_ga_sol)} / {num_features}")

    selected_idx = np.where(best_ga_sol == 1)[0]
    return selected_idx, ga_fit


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Example using Iris dataset (replace with Student Performance Dataset)
    data = load_iris()
    X = data.data
    y = data.target

    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    selected_features, best_fitness = SmartHive(X, y)

    print("\nFinal Selected Feature Indices:", selected_features)
