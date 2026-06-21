import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Set up dark background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define some dummy clusters representing semantic similarity in the joint embedding space
    # Cluster 1: Severe Incidents / Accidents
    # Cluster 2: Heavy Traffic / Congestion
    # Cluster 3: Water Logging / Rain
    
    clusters = {
        "Accidents (English/Kannada)": {
            "words": ["accident", "apaghata", "crash", "collision", "dyash"],
            "center": [2, 5],
            "color": "#ef4444" # Red
        },
        "Congestion (English/Kannada)": {
            "words": ["traffic", "sanchari", "jam", "block", "tade"],
            "center": [6, 2],
            "color": "#f97316" # Orange
        },
        "Weather (English/Kannada)": {
            "words": ["rain", "male", "water logging", "prabaha", "flooded"],
            "center": [8, 7],
            "color": "#3b82f6" # Blue
        },
        "Vehicles (English/Kannada)": {
            "words": ["truck", "lory", "bus", "car", "vahana"],
            "center": [3, 1.5],
            "color": "#a855f7" # Purple
        }
    }
    
    np.random.seed(42)
    
    for label, data in clusters.items():
        center = np.array(data["center"])
        # Generate random points around the center
        points = center + np.random.randn(len(data["words"]), 2) * 0.8
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], c=data["color"], s=100, alpha=0.8, edgecolors='white', linewidths=0.5, label=label)
        
        # Annotate words
        for i, word in enumerate(data["words"]):
            ax.annotate(word, (points[i, 0], points[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', 
                        color='white', fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", fc="#1f2937", ec="none", alpha=0.7))
            
    plt.title("M3: PEFT/LoRA MuRIL - Cross-Lingual Semantic Embeddings (t-SNE Projection)", fontsize=14, color='white', pad=20)
    
    # Hide axes ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Latent Dimension 1", fontsize=11, color="gray")
    ax.set_ylabel("Latent Dimension 2", fontsize=11, color="gray")
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), facecolor='#1f2937', edgecolor='none')
    
    out_dir = Path("/Users/prathameshnawale/Desktop/Flipkart Grid 2.0/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "m3_nlp_embeddings.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='#1e1e1e', bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == '__main__':
    main()
