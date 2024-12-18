import os
import neat
import odwroconeWahadloModelNN_modul
import odwroconeWahadloModelNN_modul_old
import visualize
import pickle
from multiprocessing import Pool
import itertools
import tkinter as tk
from tkinter import filedialog, messagebox


# Generowanie wszystkich kombinacji dla 6 wejść
xor_inputs = list(itertools.product([0, 1], repeat=6))

# Generowanie wyjść na podstawie XOR
xor_outputs = [(a ^ b ^ c ^ d, e ^ f) for (a, b, c, d, e, f) in xor_inputs]

generations = 10

# Funkcje z Twojego kodu (niezmienione)
def evaluate_genome(genome_data):
    genome_id, genome, config = genome_data
    try:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        sE = odwroconeWahadloModelNN_modul_old.odwroconeWahadloModelKx(net, False)
        fitness = -10000 - (0.05 * abs(sE[0]) + 0.05 * abs(sE[1]) + abs(sE[2]) + 0.05 * abs(sE[3]) + 0.2 * abs(sE[4]) + 0.2 * abs(sE[5]))
        return genome_id, fitness
    except Exception as e:
        print(f"Error evaluating genome {genome_id}: {e}")
        return genome_id, 0


def eval_genomes(genomes, config):
    print("Evaluating genomes with multiprocessing...")
    genome_data = [(genome_id, genome, config) for genome_id, genome in genomes]
    with Pool() as pool:
        results = pool.map(evaluate_genome, genome_data)
    for genome_id, fitness in results:
        for gid, genome in genomes:
            if gid == genome_id:
                genome.fitness = fitness
                break


def save_winner(winner, filename):
    with open(filename, 'wb') as f:
        pickle.dump(winner, f)
    print(f"Best genome saved to {filename}.")


def load_winner(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(200))
    print("Starting NEAT evolution...")
    winner = p.run(eval_genomes, generations)
    print('\nBest genome:\n{!s}'.format(winner))
    save_winner(winner, 'best_genome.pkl')
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    odwroconeWahadloModelNN_modul_old.odwroconeWahadloModelKx(net, True)
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print(f"input {xi}, expected output {xo}, got {output}")


def replay(config_file, winner_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    winner = load_winner(winner_file)
    print('\nLoaded genome:\n{!s}'.format(winner))
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    odwroconeWahadloModelNN_modul_old.odwroconeWahadloModelKx(net, True)


def resume_from_checkpoint(checkpoint_file, generations_to_run=generations):
    print(f"Restoring from checkpoint: {checkpoint_file}")
    population = neat.Checkpointer.restore_checkpoint(checkpoint_file)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(10))
    winner = population.run(eval_genomes, generations_to_run)
    print('\nBest genome after resuming:\n{!s}'.format(winner))
    return winner


# Aplikacja Tkinter
# Aplikacja Tkinter
def start_run():
    try:
        generations_to_run = int(generations_entry_run.get())
        global generations  # Ustawienie globalnej liczby generacji
        generations = generations_to_run
        run(config_path)
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid number of generations.")

def start_replay():
    winner_file = filedialog.askopenfilename(title="Select Best Genome File", filetypes=[("Pickle Files", "*.pkl")])
    if winner_file:
        replay(config_path, winner_file)

def start_resume():
    checkpoint_file = filedialog.askopenfilename(title="Select Checkpoint File", filetypes=[("Checkpoint Files", "neat-checkpoint-*")])
    if checkpoint_file:
        try:
            generations_to_run = int(generations_entry_resume.get())
            resume_from_checkpoint(checkpoint_file, generations_to_run)
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of generations.")

if __name__ == '__main__':
    # Ścieżka do pliku konfiguracyjnego
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config.txt')

    # Tworzenie okna Tkinter
    root = tk.Tk()
    root.title("NEAT Evolution Manager")

    tk.Label(root, text="NEAT Evolution Manager", font=("Arial", 16)).pack(pady=10)

    tk.Label(root, text="Run Evolution:", font=("Arial", 12)).pack(pady=5)
    generations_entry_run = tk.Entry(root)
    generations_entry_run.insert(0, "10")  # Domyślna liczba generacji
    generations_entry_run.pack(pady=5)
    tk.Button(root, text="Run Evolution", command=start_run, width=20).pack(pady=5)

    tk.Button(root, text="Replay Best Genome", command=start_replay, width=20).pack(pady=5)

    tk.Label(root, text="Resume from Checkpoint:", font=("Arial", 12)).pack(pady=5)
    generations_entry_resume = tk.Entry(root)
    generations_entry_resume.insert(0, "10")  # Domyślna liczba generacji dla resume
    generations_entry_resume.pack(pady=5)
    tk.Button(root, text="Resume", command=start_resume, width=20).pack(pady=5)

    root.mainloop()
