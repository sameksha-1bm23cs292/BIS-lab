import numpy as np
import random
import math

#obj func
def objective_function(position):
    """Booth's function: f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2. Minimum at (1, 3)."""
    x, y = position
    f = (x + 2*y - 7)**2 + (2*x + y - 5)**2
    return f
#algo
def cuckoo_search(n, pa, iterations, lower_bound, upper_bound):
    """n: Number of host nests (population size)
    pa: Probability of host bird discovering cuckoo egg 
    iterations: Maximum number of iterations
    lower_bound, upper_bound: Search space boundaries"""
    dimensions = len(lower_bound)
    # initializing search space
    nests = np.random.uniform(low=lower_bound, high=upper_bound, size=(n, dimensions))
    best_nest = nests[0]
    min_fitness = objective_function(best_nest)

    for t in range(iterations):
        #new solutions LEVY FLIGHT!
        for i in range(n):
            new_nest = get_new_nest(nests[i], lower_bound, upper_bound)
            new_fitness = objective_function(new_nest)
            #comparing fitness
            j = random.randint(0, n - 1)
            if new_fitness < objective_function(nests[j]):
                nests[j] = new_nest
            # Update the best solution
            current_best_fitness = objective_function(nests[i])
            if current_best_fitness < min_fitness:
                min_fitness = current_best_fitness
                best_nest = nests[i]
        # Abandon a fraction of the worst nests and build new ones
        nests = abandon_nests(nests, pa, lower_bound, upper_bound)
        #RANKING THE SOLUTIONS
        for i in range(n):
             current_best_fitness = objective_function(nests[i])
             if current_best_fitness < min_fitness:
                min_fitness = current_best_fitness
                best_nest = nests[i]

    return best_nest, min_fitness
def get_new_nest(nest, lower_bound, upper_bound):
    """Generate new solution using Lévy flight."""
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
             (math.gamma((1 + beta) / 2) * beta * (2**((beta - 1) / 2))))**(1 / beta)
    
    
    u = np.random.randn(len(nest)) * sigma
    v = np.random.randn(len(nest))
    step = u / abs(v)**(1 / beta)
    alpha = 0.01 
    new_nest = nest + alpha * step
    
    # Apply boundaries/constraints
    for i in range(len(new_nest)):
        if new_nest[i] < lower_bound[i]:
            new_nest[i] = lower_bound[i]
        if new_nest[i] > upper_bound[i]:
            new_nest[i] = upper_bound[i]
            
    return new_nest

def abandon_nests(nests, pa, lower_bound, upper_bound):
    """Abandon worst nests with probability pa and create new random ones."""
    n = len(nests)
    dimensions = len(lower_bound)
    for i in range(n):
        if random.random() < pa:
            # Generate new random solution
            nests[i] = np.random.uniform(low=lower_bound, high=upper_bound, size=dimensions)
    return nests

if __name__ == '__main__':
    # Optimization parameters
    NUM_NESTS = 25
    PA = 0.25
    MAX_ITERATIONS = 100
    # Define the search space bounds for x and y (e.g., -10 to 10)
    LOWER_BOUND = np.array([-10.0, -10.0])
    UPPER_BOUND = np.array([10.0, 10.0])

    # Run the Cuckoo Search algorithm
    best_position, best_fitness = cuckoo_search(
        NUM_NESTS, PA, MAX_ITERATIONS, LOWER_BOUND, UPPER_BOUND
    )
    print(f"Optimal Position (x, y): {best_position}")
    print(f"Minimum Fitness Value: {best_fitness}")
