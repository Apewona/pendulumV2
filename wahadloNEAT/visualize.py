import warnings
import matplotlib.pyplot as plt
import numpy as np


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """Plots the population's average and best fitness."""
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """Plots the trains for a single spiking neuron."""
    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(4, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, f_values, "r-")

    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(4, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig



def plot_species(statistics, view=False, filename='speciation.svg'):
    """Visualizes speciation throughout evolution."""
    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """Visualizes a neural network using matplotlib."""
    import networkx as nx

    # Create a directed graph
    graph = nx.DiGraph()

    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    if node_colors is None:
        node_colors = {}

    # Add nodes
    input_nodes = config.genome_config.input_keys
    output_nodes = config.genome_config.output_keys

    for node in input_nodes:
        graph.add_node(node_names.get(node, node), type='input', color='lightgray')

    for node in output_nodes:
        graph.add_node(node_names.get(node, node), type='output', color='lightblue')

    for node in genome.nodes:
        if node not in input_nodes and node not in output_nodes:
            graph.add_node(node_names.get(node, node), type='hidden', color='white')

    # Add edges
    for connection in genome.connections.values():
        if connection.enabled or show_disabled:
            graph.add_edge(connection.key[0], connection.key[1], weight=connection.weight)

    # Draw the graph
    pos = nx.spring_layout(graph)  # Use spring layout for visualization
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    edge_colors = ['green' if data['weight'] > 0 else 'red' for _, _, data in graph.edges(data=True)]

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=graph.nodes,
                           node_color=[node_colors.get(node, 'white') for node in graph.nodes])

    # Draw edges
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors)
    nx.draw_networkx_labels(graph, pos)

    if filename:
        plt.savefig(filename, format=fmt)

    if view:
        plt.show()

    plt.close()
