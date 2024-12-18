# NEAT Evolution Manager with Inverted Pendulum Simulation

This project implements an inverted pendulum simulation controlled by a neural network, optimized using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. It includes a graphical interface (Tkinter) for easy interaction and management.

## Features
- **Evolution Management**: Start a new NEAT evolution process.
- **Replay Best Genome**: Visualize the simulation using a previously saved genome.
- **Resume from Checkpoint**: Continue evolution from a saved checkpoint.
- **Exit Application**: Easily close the GUI.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Apewona/pendulumV2/
   ```
2. Install dependencies:
   ```bash
   pip install pygame pymunk neat-python
   ```

---

## How to Use
### Running the Application
Run the main script using Python:
```bash
python odwWahNN_Neat.py
```
This launches the graphical interface (GUI).

### GUI Instructions
#### 1. **Run Evolution**
- Enter the desired number of generations in the text box under "Run Evolution".
- Click the **Run Evolution** button to start the evolution process.

#### 2. **Replay Best Genome**
- Click the **Replay Best Genome** button.
- Select a previously saved genome file (e.g., `best_genome.pkl`).
- The simulation will visualize the performance of the selected genome.

#### 3. **Resume from Checkpoint**
- Enter the number of generations to run after resuming in the text box under "Resume from Checkpoint".
- Click the **Resume** button.
- Select a saved checkpoint file (e.g., `neat-checkpoint-*`).
- The evolution process will resume from the checkpoint.

#### 4. **Exit**
- Click the **Exit** button to close the application.

---

## Code Overview
### Main Components
1. **Simulation** (`odwroconeWahadloModelKx`):
   - Simulates the inverted pendulum controlled by a neural network.
   - Outputs cumulative error metrics for fitness evaluation.

2. **NEAT Evolution**:
   - **Run**: Starts a new evolution process.
   - **Replay**: Visualizes the best genome from a saved file.
   - **Resume**: Continues evolution from a checkpoint.

3. **GUI**:
   - Provides a user-friendly interface for managing evolution and simulations.

### Configuration File
The NEAT configuration file (`neat-config.txt`) defines parameters for the NEAT algorithm, such as population size, mutation rates, and selection criteria. Ensure this file is present in the same directory as the script.

---

## File Descriptions
- **`odwWahNN_Neat.py`**: Main script containing the GUI and logic for evolution management.
- **`odwroconeWahadloModelNN_modul_old.py`**: Contains the `odwroconeWahadloModelKx` function, which simulates the inverted pendulum.
- **`neat-config.txt`**: Configuration file for NEAT algorithm.
- **Saved Files**:
  - `best_genome.pkl`: Stores the best genome from evolution.
  - `neat-checkpoint-*`: Checkpoints for resuming evolution.

---

## Notes
1. Ensure dependencies (`pygame`, `pymunk`, `neat-python`) are installed before running.
2. Use descriptive filenames for saved genomes and checkpoints for better management.
3. The application is designed for educational purposes and demonstration of neural network optimization.

---

