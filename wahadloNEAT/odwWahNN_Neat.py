"""
2-input XOR example with multiprocessing for genome evaluation.
"""

import os
import neat
import odwroconeWahadloModelNN_modul
import visualize
from multiprocessing import Pool

# Adjusted XOR inputs and outputs to match network expectations
xor_inputs = [(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)]
xor_outputs = [(0,), (1,), (1,), (0,)]

generations = 1000
def evaluate_genome(genome_data):
    """
    Function to evaluate a single genome.
    """
    genome_id, genome, config = genome_data
    try:
        # Create a network from the genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Evaluate the network on the custom module
        sE = odwroconeWahadloModelNN_modul.odwroconeWahadloModelKx(net, False)
        
        # Compute fitness
        fitness = -10000 - (0.05 * abs(sE[0]) + 0.05 * abs(sE[1]) + abs(sE[2]) + 0.05 * abs(sE[3]))
        return genome_id, fitness
    except Exception as e:
        print(f"Error evaluating genome {genome_id}: {e}")
        return genome_id, 0  # Assign fitness 0 if an error occurs


def eval_genomes(genomes, config):
    """
    Evaluate genomes to compute fitness using multiprocessing.
    """
    print("Evaluating genomes with multiprocessing...")
    
    # Prepare data for multiprocessing
    genome_data = [(genome_id, genome, config) for genome_id, genome in genomes]

    # Use a Pool to evaluate genomes in parallel
    with Pool() as pool:
        results = pool.map(evaluate_genome, genome_data)

    # Update genome fitness based on multiprocessing results
    for genome_id, fitness in results:
        for gid, genome in genomes:
            if gid == genome_id:
                genome.fitness = fitness
                break


def run(config_file):
    """
    Run the NEAT algorithm with the given configuration file.
    """
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population
    p = neat.Population(config)

    # Add reporters to show progress
    p.add_reporter(neat.StdOutReporter(True))  # Displays generation info in the console
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)  # Keeps track of statistics
    p.add_reporter(neat.Checkpointer(200))  # Saves checkpoints every 5 generations

    # Run the NEAT algorithm
    print("Starting NEAT evolution...")
    winner = p.run(eval_genomes, generations)  # Run up to 1000 generations

    # Display the winning genome
    print('\nBest genome:\n{!s}'.format(winner))

    # Create and test the winning network
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    odwroconeWahadloModelNN_modul.odwroconeWahadloModelKx(net, True)

    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print(f"input {xi}, expected output {xo}, got {output}")

    # Check for checkpoint restoration
    checkpoint_file = 'neat-checkpoint-4'
    if os.path.exists(checkpoint_file):
        print(f"Restoring from checkpoint: {checkpoint_file}")
        restored_p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        restored_p.run(eval_genomes, 10)  # Run additional 10 generations
    else:
        print(f"Checkpoint file '{checkpoint_file}' not found. Skipping restoration.")


if __name__ == '__main__':
    # Determine path to configuration file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config.txt')
    try:
        run(config_path)
    except Exception as e:
        print(f"Failed to run NEAT: {e}")
