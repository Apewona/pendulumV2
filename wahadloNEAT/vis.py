import pickle
import os
import plotly.graph_objects as go
import neat

local_dir = os.path.dirname(__file__)

config_file = os.path.join(local_dir, 'neat-config.txt')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

with open("best_genome_resumed.pkl", "rb") as f:
    winner = pickle.load(f)

node_names = {-6: 'x[0]', -5: 'x[1]', -4: 'x[2]', -3: 'x[3]', -2: 'x[4]', -1: 'x[5]', 0: 'y[0]', 1: 'y[1]'}

def visualize_network_plotly(config, genome, node_names=None):
    # Zbiór węzłów i połączeń
    nodes = []
    edges = []
    labels = []

    # Kolory dla węzłów
    input_color = "lightblue"
    output_color = "lightgreen"
    hidden_color = "lightyellow"

    # Pozycjonowanie węzłów na osi Y
    y_positions = {
        **{key: -1 for key in config.genome_config.input_keys},  # Węzły wejściowe na dole
        **{key: 1 for key in config.genome_config.output_keys},  # Węzły wyjściowe na górze
    }

    # Dodaj węzły wejściowe i wyjściowe do listy
    for key in config.genome_config.input_keys:
        nodes.append((key, input_color))
        labels.append(node_names.get(key, str(key)))
    for key in config.genome_config.output_keys:
        nodes.append((key, output_color))
        labels.append(node_names.get(key, str(key)))

    # Dodaj ukryte węzły do listy
    hidden_nodes = set(genome.nodes.keys()) - set(config.genome_config.input_keys) - set(config.genome_config.output_keys)
    for key in hidden_nodes:
        nodes.append((key, hidden_color))
        labels.append(node_names.get(key, str(key)))
        y_positions[key] = 0  # Pozycja węzłów ukrytych na środku osi Y

    # Dodaj połączenia
    for conn_key, conn in genome.connections.items():
        if conn.enabled:
            edges.append((conn_key[0], conn_key[1], conn.weight))

    # Przygotowanie współrzędnych dla węzłów
    x_positions = {key: i for i, key in enumerate(sorted(y_positions.keys()))}
    node_x = [x_positions[key] for key, _ in nodes]
    node_y = [y_positions[key] for key, _ in nodes]

    # Rysowanie węzłów
    fig = go.Figure()
    for i, (key, color) in enumerate(nodes):
        fig.add_trace(go.Scatter(
            x=[node_x[i]],
            y=[node_y[i]],
            mode='markers+text',
            marker=dict(size=20, color=color, line=dict(width=2, color='black')),
            text=labels[i],
            textposition='top center',
            name=f'Node {key}'
        ))

    # Rysowanie połączeń
    for src, dst, weight in edges:
        x_start, x_end = x_positions[src], x_positions[dst]
        y_start, y_end = y_positions[src], y_positions[dst]
        fig.add_trace(go.Scatter(
            x=[x_start, x_end, None],
            y=[y_start, y_end, None],
            mode='lines',
            line=dict(color='green' if weight > 0 else 'red', width=abs(weight) * 2),
            hoverinfo='none'
        ))

    # Ustawienia wykresu
    fig.update_layout(
        title="Wizualizacja Sieci NEAT",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
    )

    fig.show()

# Użycie funkcji
visualize_network_plotly(config, winner, node_names)
