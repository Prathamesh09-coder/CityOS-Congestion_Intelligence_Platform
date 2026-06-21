import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path

def main():
    # Set up dark background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create a realistic-looking road network graph
    # We use a random geometric graph to simulate road intersections
    np.random.seed(42)
    G = nx.random_geometric_graph(120, radius=0.15, seed=42)
    
    # Select a "source" node for an accident/closure
    source_node = 45
    
    # Calculate shortest path lengths from source to simulate cascading delay
    lengths = nx.single_source_shortest_path_length(G, source_node)
    
    # Define node colors and sizes based on distance to the closure
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        if node == source_node:
            node_colors.append('#ef4444') # Red (Critical Congestion / Closure)
            node_sizes.append(400)
        elif node in lengths:
            dist = lengths[node]
            if dist == 1:
                node_colors.append('#f97316') # Orange (Heavy Delay)
                node_sizes.append(200)
            elif dist == 2:
                node_colors.append('#eab308') # Yellow (Moderate Delay)
                node_sizes.append(100)
            elif dist == 3:
                node_colors.append('#3b82f6') # Blue (Slight Delay - WaveNet propagation)
                node_sizes.append(50)
            else:
                node_colors.append('#1f2937') # Dark Gray (Normal Flow)
                node_sizes.append(30)
        else:
            node_colors.append('#1f2937')
            node_sizes.append(30)
            
    # Get positions for layout
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#374151', alpha=0.6, width=1.0)
    
    # Highlight edges propagating from the source to simulate wave/cascading delays
    edge_colors = []
    for u, v in G.edges():
        if (u == source_node and lengths.get(v) == 1) or (v == source_node and lengths.get(u) == 1):
            edge_colors.append('#ef4444')
        elif (lengths.get(u) == 1 and lengths.get(v) == 2) or (lengths.get(v) == 1 and lengths.get(u) == 2):
            edge_colors.append('#f97316')
        elif (lengths.get(u) == 2 and lengths.get(v) == 3) or (lengths.get(v) == 2 and lengths.get(u) == 3):
            edge_colors.append('#eab308')
        else:
            edge_colors.append('none')
            
    # Draw highlighted edges
    highlighted_edges = [(u, v) for i, (u, v) in enumerate(G.edges()) if edge_colors[i] != 'none']
    h_colors = [c for c in edge_colors if c != 'none']
    if highlighted_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=highlighted_edges, edge_color=h_colors, width=2.5, alpha=0.9)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.9, edgecolors='white', linewidths=0.5)
    
    # Annotate source
    ax.annotate("Incident Source\n(Graph WaveNet Spatial Origin)",
                xy=pos[source_node], xycoords='data',
                xytext=(pos[source_node][0] + 0.05, pos[source_node][1] + 0.1), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
                color='white', fontsize=10, fontweight='bold', ha='left')
    
    plt.title("M4: Spatio-Temporal Graph WaveNet - Cascading Delays Simulation", fontsize=14, color='white', pad=20)
    plt.axis('off')
    
    out_dir = Path("/Users/prathameshnawale/Desktop/Flipkart Grid 2.0/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "m4_gnn_cascading.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='#1e1e1e', bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == '__main__':
    main()
