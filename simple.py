import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import random, math, copy
from tqdm import tqdm

# for visualization
from utils import * 

import warnings
warnings.filterwarnings("ignore")

class Environment:
    def __init__(self):
        self.MAX_ITER = 1000
        self.THRERESHOLD = 10 

    def reset(self, user_func_1, user_func_2):
        self.x1_values = np.zeros(self.MAX_ITER)
        self.x2_values = np.zeros(self.MAX_ITER)
        self.time_steps = np.arange(self.MAX_ITER)
        self.packages_lost = 0
        self.percent_utilization = []

        self.x1 = 1
        self.x2 = 2

        self.uf_1 = user_func_1
        self.uf_2 = user_func_2

    def simulate(self):
        for i in range(self.MAX_ITER):
            if (self.x1 + self.x2 <= self.THRERESHOLD):
                self.percent_utilization.append((self.x1 + self.x2) / self.THRERESHOLD)
                self.x1 = self.uf_1.call_alpha(self.x1)
                self.x2 = self.uf_2.call_alpha(self.x2)
            else:
                self.x1 = self.uf_1.call_beta(self.x1)
                self.x2 = self.uf_2.call_beta(self.x2)
                self.percent_utilization.append(0)
                self.packages_lost += 2

            self.x1_values[i] = self.x1
            self.x2_values[i] = self.x2
            

    def assess(self):
        # plot the results
        self.plot_results()
        avg_utilization, packages_lost = self.calculate_score()
        print("Final Results")
        print("Average Utilization:", avg_utilization)
        print("Packages Lost:", packages_lost)

    def calculate_score(self):
        mean_ = np.mean(self.percent_utilization)
        # if mean is NaN or Inf then return 0
        if np.isnan(mean_) or np.isinf(mean_):
            mean_ = 0
        return mean_, self.packages_lost



    def plot_results(self):
        # 3D Plot
        fig = plt.figure(figsize=(14, 6))

        points = np.array([self.x1_values, self.x2_values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(self.time_steps.min(), self.time_steps.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2)
        lc.set_array(self.time_steps)

        # 3D plot with color coding
        ax1 = fig.add_subplot(121, projection='3d')

        # Create a color map based on the time steps for the 3D line
        colors = plt.get_cmap('viridis')(norm(self.time_steps))

        # Plot each segment with a color corresponding to its time step
        for i in range(len(segments)):
            ax1.plot3D(segments[i][:, 0], segments[i][:, 1], [self.time_steps[i], self.time_steps[i+1]], color=colors[i])

        ax1.set_title('3D Plot of x1 and x2 over Time with Color Gradient')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_zlabel('Time')

        # Second subplot for gradient line plot
        ax2 = fig.add_subplot(122)
        ax2.add_collection(lc)
        ax2.autoscale()
        fig.colorbar(lc, ax=ax2).set_label('Time Step')
        ax2.set_title('Line Plot of x1 and x2 with Time Gradient')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')

        #plt.tight_layout()
        plt.show()




# GA relevant classes

# Define operations
def add(x, y): return x + y
def add_three(x, y, z): return x + y + z
def add_four(x, y, z, w): return x + y + z + w
def add_five(x, y, z, w, v): return x + y + z + w + v
def sub(x, y): return x - y
def mul(x, y): return x * y
def mul_three(x, y, z): return x * y * z
def mul_four(x, y, z, w): return x * y * z * w
def mul_five(x, y, z, w, v): return x * y * z * w * v
def div(x, y): return x / y if y != 0 else 1
def sin(x): return math.sin(x)
def exp(x): return math.exp(x)
def log(x): return math.log(x) if x > 0 else 0

operations = [add, sub, mul, div, sin, exp, log, add_three, add_four, add_five, mul_three, mul_four, mul_five]
variables = ['x']


class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []
    
    def evaluate(self, env_x):
        if self.value in variables:
            return env_x 
        elif callable(self.value):
            args = [child.evaluate(env_x) for child in self.children]
            return self.value(*args)
        else:  # Parameter or constant
            return self.value
        
    def copy(self):
        return Node(self.value, [child.copy() for child in self.children])
    
def create_random_tree(max_depth=5, depth=0):
    # check if max_depth is reached
    if depth >= max_depth:
        # randomly decide whether to create a variable or a constant
        if random.random() < 0.5:
            return Node(random.choice(variables))
        else:
            return Node(random.uniform(-1, 1))
    # at each iteration, randomly decide whether to go deeper for each one.
    # First, decide whether to create a variable, a constant or a function
    rndm_num = random.random()
    if rndm_num < 0.2:
        # Create a variable
        return Node(random.choice(variables))
    elif rndm_num < 0.4:
        # Create a constant
        return Node(random.uniform(-1, 1))
    else:
        # Create a function
        op = random.choice(operations)
        args_count = op.__code__.co_argcount
        children = [create_random_tree(max_depth, depth+1) for _ in range(args_count)]
        return Node(op, children)


def create_population(population_size):
    return [[create_random_tree(), create_random_tree()] for _ in range(population_size)]

def print_as_equation(node):
    if node.value in variables:
        return node.value
    elif callable(node.value):
        args = [print_as_equation(child) for child in node.children]
        if node.value in [add, add_three, add_four, add_five]:
            return f"({args[0]} + {args[1]})"
        elif node.value in [sub]:
            return f"({args[0]} - {args[1]})"
        elif node.value in [mul, mul_three, mul_four, mul_five]:
            return f"({args[0]} * {args[1]})"
        elif node.value in [div]:
            return f"({args[0]} / {args[1]})"
        elif node.value in [sin, exp, log]:
            return f"{node.value.__name__}({args[0]})"
    else:  # Parameter or constant
        return str(f"{node.value:.5f}")


def mutate(tree):
    """
    Generally speaking, the mutations taken are:
    - adjusting constants by a small percentage
    - changing functions to other functions
    """
    new_tree = tree.copy()
    stack = [new_tree]
    while stack:
        node = stack.pop()
        if node.value in variables:
            continue
        elif callable(node.value):
            if random.random() < 0.1:  # 10% chance of changing the function
                # change the function to another with the same number of accepted inputs
                func_subset = [op for op in operations if op.__code__.co_argcount == node.value.__code__.co_argcount]
                node.value = random.choice(func_subset)
        else:  # Parameter or constant
            if random.random() < 0.1:  # 10% chance of changing the constant
                node.value += random.uniform(-0.05, 0.05)
        stack.extend(node.children)
    return new_tree

def unravel_tree(tree):
    """
    Unravel the tree into a list of nodes.
    """
    nodes = []
    stack = [tree]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children)
    return nodes

def crossover(tree1, tree2, max_depth=5):
    """
    At each Node, for all sub-node, decide whether to take the sub-node from tree1 or tree2.
    """
    new_tree = tree1.copy()
    stack = [new_tree]
    choices = unravel_tree(tree2)
    depth = 0
    while stack:
        node = stack.pop()
        if random.random() < 0.1:
            # select random from choices
            random_subtree = random.choice(choices)
            node.value = random_subtree.value
            node.children = random_subtree.children
        stack.extend(node.children)
        choices.extend(node.children)
        depth += 1
        if depth > max_depth:
            # add leaf nodes and break
            for node in stack:
                # check if already leaf node
                if callable(node.value):
                    node.children = [create_random_tree(depth=9999) for _ in range(node.value.__code__.co_argcount)]
            break
    return new_tree

def create_new_population(population, fitness_scores, population_size):
    """
    Create a new population based on the fitness scores
        - keep the top 20% of the population
        - select 20% of the population for mutation
        - fill 40% of the population with crossover
        - fill 20% of the population with random trees
    """
    new_population = []
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]

    # fill 20% with the best from the previous run
    new_population.extend(copy.deepcopy(sorted_population[:int(population_size * 0.2)]))

    # fill 20% with mutation (keeping in mind that we have two trees per individual)
    for individual in sorted_population[:int(population_size * 0.2)]:
        new_population.append([mutate(copy.deepcopy(individual[0])), mutate(copy.deepcopy(individual[1]))])

    # fill 40% with crossover (keeping in mind that we have two trees per individual)
    for _ in range(int(population_size * 0.4)):
        parent1, parent2 = random.sample(sorted_population, 2)
        new_population.append([
            crossover(copy.deepcopy(parent1[0]), copy.deepcopy(parent2[0])),
            crossover(copy.deepcopy(parent1[1]), copy.deepcopy(parent2[1]))
        ])

    # fill 20% with random trees
    for _ in range(int(population_size * 0.2)):
        new_population.append([create_random_tree(), create_random_tree()])
    
    return new_population

class FunctionClass:
    def __init__(self, alpha_function, beta_function):
        self.alpha_function = alpha_function
        self.beta_function = beta_function
    
    def call_alpha(self, x):
        return np.max([self.alpha_function.evaluate(x), 0])
    
    def call_beta(self, x):
        return np.max([self.beta_function.evaluate(x), 0])
    
def get_fitness_scores(population, environment):
    """
    Simply iterate over the population, transform the tree into a function and run the simulation.
    """
    fitness_scores = [] 
    #for individual in population:
    for individual in tqdm(population, desc='Evaluating Fitness', leave=False):
        # convert to two different function classes
        user_func_1 = FunctionClass(individual[0], individual[1])
        user_func_2 = FunctionClass(individual[0], individual[1])
        environment.reset(user_func_1, user_func_2)
        try:
            environment.simulate()
            pct_util, packages_lost = environment.calculate_score()
            fitness_scores.append(pct_util)
        except Exception as e:
            #print(f"Failed with error: {e}")
            fitness_scores.append(0)
    return fitness_scores


# actually run the GA
EPOCHS = 100
population_size = 100
environment = Environment()
population = create_population(population_size)

# Lists to store statistics
top_20_fitness_history = []
max_fitness_history = []
min_fitness_history = []
mean_fitness_history = []
std_fitness = []
pct_failed = []

# Use TQDM for progress tracking
for epoch in tqdm(range(EPOCHS), desc='GA Progress'):
    fitness_scores = get_fitness_scores(population, environment)
    
    # Updating histories
    sorted_scores = sorted(fitness_scores, reverse=True)
    top_20_fitness_history.append(sorted_scores[:20])
    max_fitness = max(fitness_scores)
    min_fitness = min(fitness_scores)
    std_fitness = np.std(fitness_scores)
    mean_fitness = np.mean(fitness_scores)
    max_fitness_history.append(max_fitness)
    min_fitness_history.append(min_fitness)
    mean_fitness_history.append(mean_fitness)
    pct_failed.append(fitness_scores.count(0) / population_size)
    
    # Update tqdm description with statistics
    tqdm.write(f"Epoch {epoch}: Max Fitness = {max_fitness}, Min Fitness = {min_fitness}, Mean Fitness = {mean_fitness}, Failed = {pct_failed[-1]:.2%}")
    # Create new population for the next epoch
    population = create_new_population(population, fitness_scores, population_size)

# plot it
plt.figure(figsize=(10, 6))

# Plotting Max, Min, and Mean Fitness
plt.plot(max_fitness_history, label='Max Fitness', linewidth=2)
plt.plot(min_fitness_history, label='Min Fitness', linewidth=2)
plt.fill_between(range(EPOCHS), mean_fitness_history - std_fitness, mean_fitness_history + std_fitness, color='gray', alpha=0.2)
plt.plot(mean_fitness_history, label='Mean Fitness', color='black', linestyle='--', linewidth=2)

# Adding error bars to mean fitness plot
# plt.errorbar(range(epochs), mean_fitness_history, yerr=std_fitness, label='Mean Fitness', fmt='-o')

# Enhancing the plot
plt.title('Genetic Algorithm Performance Over Epochs', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Fitness', fontsize=14)
plt.ylim(0, 1)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

# Show the improved plot
plt.show()


# get the best 10 individuals
best_individuals = sorted(population, key=lambda x: get_fitness_scores([x], environment)[0], reverse=True)[:10]

# store each one as pkl
import pickle
for i, individual in enumerate(best_individuals):
    with open(f'best_individual_{i}.pkl', 'wb') as f:
        pickle.dump(individual, f)

# run the best one and visualize the results
best_individual = best_individuals[0]
user_func_1 = FunctionClass(best_individual[0], best_individual[1])
user_func_2 = FunctionClass(best_individual[0], best_individual[1])
environment.reset(user_func_1, user_func_2)
environment.simulate()
environment.assess()

# print both equations
print("Equation 1:", print_as_equation(best_individual[0]))
print("Equation 2:", print_as_equation(best_individual[1]))

# visualize the tree
visualize_tree(best_individual[0])
visualize_tree(best_individual[1])



