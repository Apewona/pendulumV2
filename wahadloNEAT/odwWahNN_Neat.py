import os
import neat
import odwroconeWahadloModelNN_modul
import odwroconeWahadloModelNN_modul_old
import visualize
import pickle
from multiprocessing import Pool
import itertools

# Generowanie wszystkich kombinacji dla 6 wejść
xor_inputs = list(itertools.product([0, 1], repeat=6))

# Generowanie wyjść na podstawie XOR
xor_outputs = [(a ^ b ^ c ^ d, e ^ f) for (a, b, c, d, e, f) in xor_inputs]

generations = 10

def evaluate_genome(genome_data):
    """
    Function to evaluate a single genome.
    """
    genome_id, genome, config = genome_data
    try:
        # Create a network from the genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Evaluate the network on the custom module
        sE = odwroconeWahadloModelNN_modul_old.odwroconeWahadloModelKx(net, False)
        
        # Compute fitness
        # x=(cartX, cartVX, cartA, cartVA, arm2_A, arm2_VA)
        fitness = -10000 - (0.05 * abs(sE[0]) + 0.05 * abs(sE[1]) + abs(sE[2]) + 0.05 * abs(sE[3]) + 0.2 * abs(sE[4]) + 0.2 * abs(sE[5]))
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


def save_winner(winner, filename):
    """
    Save the best genome to a file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(winner, f)
    print(f"Best genome saved to {filename}.")


def load_winner(filename):
    """
    Load the best genome from a file.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


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

    # Save the best genome
    save_winner(winner, 'best_genome.pkl')

    # Create and test the winning network
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    odwroconeWahadloModelNN_modul_old.odwroconeWahadloModelKx(net, True)

    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print(f"input {xi}, expected output {xo}, got {output}")


def replay(config_file, winner_file):
    """
    Replay a saved genome.
    """
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Load the best genome
    winner = load_winner(winner_file)

    # Display the loaded genome
    print('\nLoaded genome:\n{!s}'.format(winner))

    # Create and test the network
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    odwroconeWahadloModelNN_modul_old.odwroconeWahadloModelKx(net, True)

def resume_from_checkpoint(checkpoint_file, generations_to_run=generations):
    """
    Resume NEAT training from a given checkpoint.
    """
    # Odtwórz populację z checkpointu
    print(f"Restoring from checkpoint: {checkpoint_file}")
    population = neat.Checkpointer.restore_checkpoint(checkpoint_file)

    # Dodaj reporterów, jeśli chcesz zobaczyć statystyki
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(10))  # Nadal zapisuj nowe checkpointy

    # Kontynuuj proces uczenia przez określoną liczbę generacji
    winner = population.run(eval_genomes, generations_to_run)

    # Wyświetl najlepszego osobnika
    print('\nBest genome after resuming:\n{!s}'.format(winner))
    return winner

if __name__ == '__main__':
    # Determine path to configuration file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config.txt')

    try:
        # Run evolution, replay genome, or resume from checkpoint
        mode = input("Enter 'run', 'replay', or 'resume': ").strip().lower()
        if mode == 'run':
            run(config_path)
        elif mode == 'replay':
            replay(config_path, 'best_genome.pkl')
        elif mode == 'resume':
            # Prompt for checkpoint file name
            checkpoint_file = input("Enter the checkpoint file name (e.g., 'neat-checkpoint-4'): ").strip()
            if os.path.exists(checkpoint_file):
                generations_to_run = int(input("Enter the number of generations to run: ").strip())
                resume_from_checkpoint(checkpoint_file, generations_to_run)
            else:
                print(f"Checkpoint file '{checkpoint_file}' not found.")
        else:
            print("Invalid mode. Please enter 'run', 'replay', or 'resume'.")
    except Exception as e:
        print(f"Failed to run NEAT: {e}")

