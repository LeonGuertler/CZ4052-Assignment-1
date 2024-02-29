import random, time, math, copy, pickle, os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


from concurrent.futures import ProcessPoolExecutor, as_completed


# set seeds
random.seed(489)
np.random.seed(489)


class Network:
    def __init__(self, capacity):
        self.capacity = capacity
        self.current_load = 0
        self.network_utilization = []
        self.packages_lost = 0
        self.individuals = 0
    
    def check_capacity(self) -> bool:
        if self.current_load > self.capacity:
            self.network_utilization.append(0)
            self.packages_lost += self.individuals
            return False
        else:
            self.network_utilization.append((self.current_load / self.capacity) * 100)
            return True
    
    def add_load(self, packet_size):
        self.current_load += packet_size
        self.individuals += 1

    def reset_load(self):
        self.current_load = 0
        self.individuals = 0

    def get_average_utilization(self):
        mean_ = np.mean(self.network_utilization)
        if np.isnan(mean_) or np.isinf(mean_):
            return 0
        return mean_
    
    def packages_lost(self):
        return self.packages_lost






class Package:
    def __init__(self, size:int):
        self.size = size
        self.active = True

    def send_partial(self, rate:int):
        self.size -= rate
        return self.size <= 0
    

class Sender:
    def __init__(self, network, flow_control, start_rate, min_packet_size, max_packet_size, prob_package):
        self.network = network
        self.flow_control = flow_control
        self.rate = start_rate
        self.prob_package = prob_package
        self.min_packet_size = min_packet_size
        self.max_packet_size = max_packet_size
        self.active_package = None 
        self.stats = {'packets_sent': [], 'packets_lost': [], 'total_size_sent': [], 'load_percentage': []}

    def try_send(self):
        # check if there is an active package 
        if self.active_package is None:
            # with prob_package, send a package
            if random.random() < self.prob_package:
                self.active_package = Package(
                    size=random.randint(self.min_packet_size, self.max_packet_size)
                )
        
        if self.active_package is not None:
            # this means there is still an active package to send
            # check if the package has "rate" data points left to be sent
            self.sending_size = np.min([self.rate, self.active_package.size])

            self.network.add_load(self.sending_size)
        
        else:
            self.sending_size = 0


    def update(self, sending_success):
        if self.sending_size == 0:
            # this means no package was attempted to be sent 
            self.stats['packets_sent'].append(0)
            self.stats['total_size_sent'].append(0)
            self.stats['packets_lost'].append(0)
            self.stats['load_percentage'].append(0)
        else:
            if sending_success:
                #self.network.add_load(self.sending_size)
                self.stats['packets_sent'].append(1)
                self.stats['total_size_sent'].append(self.sending_size)
                self.stats['packets_lost'].append(0)
                self.stats['load_percentage'].append((self.sending_size / self.network.capacity) * 100)
                if self.active_package.send_partial(self.sending_size):
                    self.active_package = None
                
            else:
                # this means the package was lost
                self.stats['packets_lost'].append(1)
                self.stats['packets_sent'].append(0)
                self.stats['total_size_sent'].append(0)
                self.stats['load_percentage'].append(0)

            # update flow parameters
            self.rate = self.flow_control.adjust_rate(sending_success, self.rate)




def simulate_network(individual):
    network_capacity = 500
    duration = 750 #1_000
    network = Network(network_capacity)
    number_senders = np.random.randint(5, 15) 
    senders = []
    start_time = time.time()
    for _ in range(number_senders):
        senders.append(
            Sender(
                network=network,
                flow_control=FlowControl(
                    alpha_function=individual[0], 
                    beta_function=individual[1],
                    c1_update_func=individual[2],
                    c2_update_func=individual[3],
                    c3_update_func=individual[4],
                    c1_update_func_neg=individual[5],
                    c2_update_func_neg=individual[6],
                    c3_update_func_neg=individual[7],
                ),
                start_rate=np.random.randint(1, 3),
                min_packet_size=np.random.randint(10, 100),
                max_packet_size=np.random.randint(100, 5_000),
                prob_package=np.random.random(),
            )
        )
    for _ in tqdm(range(duration), desc='Simulating Network', leave=False):
        
        # reset the load 
        network.reset_load()
        for sender in senders:
            sender.try_send()
            if time.time() - start_time > 15:
                print('out of time')
                return 0, 0

        # check if the network is overloaded
        success = network.check_capacity()
        for sender in senders:
            sender.update(success)
            if time.time() - start_time > 15:
                print('out of time')
                return 0, 0

        
    return network.get_average_utilization(), network.packages_lost








def plot_statistics(senders, duration):
    time_steps = list(range(duration))
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    # populate colors with gradients or viridis
    colors = plt.cm.viridis(np.linspace(0, 1, len(senders)))
    # shuffle the colors
    np.random.shuffle(colors)
    
    for i, stat in enumerate(['packets_sent', 'packets_lost', 'total_size_sent']):
        for sender_idx, sender in enumerate(senders):
            cumulative_stat = np.cumsum(sender.stats[stat])
            axs[i//2, i%2].plot(time_steps, cumulative_stat, label=f'Sender {sender_idx+1}', color=colors[sender_idx])
        axs[i//2, i%2].set_title(stat.replace('_', ' ').capitalize())

    # Stacked plot for capacity utilization
    pct_util = np.array([s.stats['load_percentage'] for s in senders])
    # remove those x-pos where all are 0
    time_steps = np.array(time_steps)[np.sum(pct_util, axis=0) > 0]
    pct_util = pct_util[:,np.sum(pct_util, axis=0) > 0]

    axs[1, 1].stackplot(
        time_steps, 
        pct_util, #[s.stats['load_percentage'] for s in senders], 
        labels=[f'Sender {i+1}' for i in range(len(senders))], 
        colors=colors
    )
    axs[1, 1].set_title('Network Capacity Utilization (%)')

    # print final statistics summary
    num_senders = len(senders)
    total_packages_sent = np.sum([np.sum(s.stats['packets_sent']) for s in senders])
    total_packages_lost = np.sum([np.sum(s.stats['packets_lost']) for s in senders])
    total_size_sent = np.sum([np.sum(s.stats['total_size_sent']) for s in senders])
    average_utilization = np.sum([np.mean(s.stats['load_percentage']) for s in senders])


    print("Final Statistics")
    print(f"Total Packages Sent: {total_packages_sent}")
    print(f"Total Packages Lost: {total_packages_lost}")
    print(f"Total Size Sent: {total_size_sent}")
    print(f"Average Utilization: {average_utilization}")



    plt.tight_layout()
    plt.show()


class FlowControl:
    def __init__(
            self, 
            alpha_function, 
            beta_function, 
            c1_update_func, 
            c2_update_func, 
            c3_update_func,
            c1_update_func_neg,
            c2_update_func_neg,
            c3_update_func_neg,
        ):
        self.alpha_function = alpha_function
        self.beta_function = beta_function
        self.additional_parameter_1 = 1
        self.additional_parameter_2 = 1
        self.additional_parameter_3 = 1
        self.additional_parameter_1_update = c1_update_func
        self.additional_parameter_2_update = c2_update_func
        self.additional_parameter_3_update = c3_update_func
        self.additional_parameter_1_update_neg = c1_update_func_neg
        self.additional_parameter_2_update_neg = c2_update_func_neg
        self.additional_parameter_3_update_neg = c3_update_func_neg

    def adjust_rate(self, success, current_rate):
        if success:
            return_val = np.max([
                self.alpha_function.evaluate(
                    current_rate, 
                    self.additional_parameter_1,
                    self.additional_parameter_2,
                    self.additional_parameter_3,
                ),
                0
            ])
            self.additional_parameter_1 = self.additional_parameter_1_update.evaluate(
                current_rate, 
                self.additional_parameter_1,
                self.additional_parameter_2,
                self.additional_parameter_3,
            )

            self.additional_parameter_2 = self.additional_parameter_2_update.evaluate(
                current_rate, 
                self.additional_parameter_1,
                self.additional_parameter_2,
                self.additional_parameter_3,
            )
            
            self.additional_parameter_3 = self.additional_parameter_3_update.evaluate(
                current_rate, 
                self.additional_parameter_1,
                self.additional_parameter_2,
                self.additional_parameter_3,
            )

            return return_val
        else:
            return_val = np.max([
                self.beta_function.evaluate(
                    current_rate, 
                    self.additional_parameter_1,
                    self.additional_parameter_2,
                    self.additional_parameter_3,
                ),
                0
            ])
            # update the parameters
            self.additional_parameter_1 = self.additional_parameter_1_update_neg.evaluate(
                current_rate, 
                self.additional_parameter_1,
                self.additional_parameter_2,
                self.additional_parameter_3,
            )

            self.additional_parameter_2 = self.additional_parameter_2_update_neg.evaluate(
                current_rate, 
                self.additional_parameter_1,
                self.additional_parameter_2,
                self.additional_parameter_3,
            )

            self.additional_parameter_3 = self.additional_parameter_3_update_neg.evaluate(
                current_rate, 
                self.additional_parameter_1,
                self.additional_parameter_2,
                self.additional_parameter_3,
            )
            return return_val
        

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
def div(x, y): return x / (y+1e-7) if y != 0 else 1
def sin(x): return math.sin(x)
def exp(x): 
    try:
        return math.exp(x)
    except:
        return 1
def log(x): return math.log(x) if x > 0 else 0

operations = [add, sub, mul, div, sin, exp, log, add_three, add_four, add_five, mul_three, mul_four, mul_five]
variables = ['x']
learned_variables = ['c1', 'c2', 'c3']


class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []
    
    def evaluate(self, env_x, c1, c2, c3):
        #print(env_x, c1, c2, c3,)
        #input()
        if self.value in variables or self.value in learned_variables:
            if self.value == 'x':
                return env_x
            elif self.value == 'c1':
                return c1
            elif self.value == 'c2':
                return c2
            elif self.value == 'c3':
                return c3
        elif callable(self.value):
            args = [child.evaluate(env_x, c1, c2, c3) for child in self.children]
            return self.value(*args)
        else:  # Parameter or constant
            return self.value
        
    def copy(self):
        return Node(self.value, [child.copy() for child in self.children])
    
def create_random_tree(max_depth=10, depth=0):
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
    if rndm_num < 0.3:
        # Create a variable
        return Node(random.choice(variables))
    if rndm_num < 0.4:
        # Add a learned variable
        return Node(random.choice(learned_variables))
    elif rndm_num < 0.5:
        # Create a constant
        return Node(random.uniform(-1, 1))
    else:
        # Create a function
        op = random.choice(operations)
        args_count = op.__code__.co_argcount
        children = [create_random_tree(max_depth, depth+1) for _ in range(args_count)]
        return Node(op, children)


def create_population(population_size):
    return [[create_random_tree() for __ in range(8)] for _ in range(population_size)]

def print_as_equation(node):
    if node.value in variables:
        return node.value
    if node.value in learned_variables:
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
        if node.value in variables or node.value in learned_variables:
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
    print("\n\nCreating new population\n\n")
    new_population = []
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]

    # fill 20% with the best from the previous run
    new_population.extend(copy.deepcopy(sorted_population[:int(population_size * 0.2)]))

    # fill 20% with mutation (keeping in mind that we have two trees per individual)
    for individual in sorted_population[:int(population_size * 0.2)]:
        new_gene = []
        for gene in individual:
            new_gene.append(mutate(copy.deepcopy(gene)))
        new_population.append(new_gene)

    # fill 40% with crossover (keeping in mind that we have two trees per individual)
    for _ in range(int(population_size * 0.4)):
        parent1, parent2 = random.sample(sorted_population, 2)
        new_gene = []
        for gene1, gene2 in zip(parent1, parent2):
            new_gene.append(crossover(copy.deepcopy(gene1), copy.deepcopy(gene2)))
        new_population.append(new_gene)

    # fill 20% with random trees
    for _ in range(int(population_size * 0.2)):
        new_population.append([create_random_tree() for __ in range(8)])
    
    return new_population



def evaluate_individual(individual, iters=10, idx=0):
    fitness_list = []
    for _ in range(iters):
        try:
            fitness, _ = simulate_network(individual)
        except:
            fitness = 0
        fitness_list.append(fitness)
    return np.mean(fitness_list), idx


def get_fitness_scores(population, iters=10):
    with ProcessPoolExecutor() as executor:
        # Submit all tasks and create a list of futures
        futures = [executor.submit(evaluate_individual, ind, iters, idx) for idx, ind in enumerate(population)]
        
        # Initialize an empty list to store results
        fitness_scores = np.zeros(len(population))
        
        # Iterate over the futures as they complete
        for future in tqdm(as_completed(futures), total=len(population), desc='Calculating Fitness'):
            # Extract the result from each completed future
            result, idx = future.result()
            fitness_scores[idx] = result
            #fitness_scores.append(result)
    
    return list(fitness_scores)

def save_population(population, fitness_scores, epoch):
    folder_name = "saved_population"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    checkpoint = {
        'epoch': epoch,
        'population': population,
        'fitness_scores': fitness_scores,
    }
    with open(f'saved_population/checkpoint_epoch_{epoch}.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(filename):
    with open(os.path.join("saved_checkpoint", filename), 'rb') as f:
        checkpoint = pickle.read(f)

    epoch = checkpoint['epoch']
    population = checkpoint['population']
    fitness_scores = checkpoint['fitness_scores']

    return epoch, population, fitness_scores


# actually run the GA
EPOCHS = 50
population_size = 250
population = create_population(population_size)

# Lists to store statistics
top_20_fitness_history = []
max_fitness_history = []
min_fitness_history = []
mean_fitness_history = []
std_fitness = []
pct_failed = []

load_checkpoint = False

if load_checkpoint:
    _, population, _ = load_checkpoint()

# Use TQDM for progress tracking
for epoch in tqdm(range(EPOCHS), desc='GA Progress'):
    fitness_scores = get_fitness_scores(population)

    # save the current population
    save_population(population, fitness_scores, epoch)

    # print best gene
    best_individual = population[np.argmax(fitness_scores)]
    print("alpha update:", print_as_equation(best_individual[0]))
    print("beta update:", print_as_equation(best_individual[1]))
    print("c1:", print_as_equation(best_individual[2]))
    print("c2:", print_as_equation(best_individual[3]))
    print("c3:", print_as_equation(best_individual[4]))
    print("c1 update:", print_as_equation(best_individual[5]))
    print("c2 update:", print_as_equation(best_individual[6]))
    print("c3 update:", print_as_equation(best_individual[7]))
    
    # Updating histories
    sorted_scores = sorted(fitness_scores, reverse=True)
    top_20_fitness_history.append(sorted_scores[:20])
    max_fitness = max(sorted_scores)
    min_fitness = min(sorted_scores)
    mean_fitness = np.mean(sorted_scores)
    std_fitness = np.std(sorted_scores)
    max_fitness_history.append(max_fitness)
    min_fitness_history.append(min_fitness)
    mean_fitness_history.append(mean_fitness)
    pct_failed.append(fitness_scores.count(0) / len(fitness_scores))
    
    # Update tqdm description with statistics
    tqdm.write(f"Epoch {epoch}: Max Fitness = {max_fitness}, Min Fitness = {min_fitness}, Mean Fitness = {mean_fitness}, Failed = {pct_failed[-1]:.2%}")
    # Create new population for the next epoch
    population = create_new_population(population, fitness_scores, population_size)



# get the best 10 individuals
best_individuals = sorted(population, key=lambda x: fitness_scores, reverse=True)[:10]
# run the best one and visualize the results
best_individual = best_individuals[0]
fitness, packages_lost = simulate_network(best_individual)

# print both equations
print("alpha update:", print_as_equation(best_individual[0]))
print("beta update:", print_as_equation(best_individual[1]))
print("c1:", print_as_equation(best_individual[2]))
print("c2:", print_as_equation(best_individual[3]))
print("c3:", print_as_equation(best_individual[4]))
print("c1 update:", print_as_equation(best_individual[5]))
print("c2 update:", print_as_equation(best_individual[6]))
print("c3 update:", print_as_equation(best_individual[7]))

# plot the statistics
print(fitness, packages_lost)