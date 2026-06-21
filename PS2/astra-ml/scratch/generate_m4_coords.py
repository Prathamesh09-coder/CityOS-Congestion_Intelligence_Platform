import json
import logging
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from astra_ml.data.road_graph import get_or_build_road_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_m4_coords")

def generate():
    models_dir = Path("models")
    m4_path = models_dir / "m4_model.pth"
    if not m4_path.exists():
        logger.error(f"m4_model.pth not found in {models_dir}")
        return

    logger.info("Loading m4_model.pth...")
    checkpoint = torch.load(m4_path, map_location="cpu")
    nodes = checkpoint.get("nodes", [])
    logger.info(f"Loaded GNN checkpoint with {len(nodes)} nodes.")

    logger.info("Loading road graph...")
    road_graph = get_or_build_road_graph()

    coords_map = {}
    missing_count = 0

    for idx, node_id in enumerate(nodes):
        node_key = node_id
        if not road_graph.has_node(node_key):
            try:
                node_key = int(node_id)
            except ValueError:
                pass
        
        if road_graph.has_node(node_key):
            node_data = road_graph.nodes[node_key]
            lat_val = node_data.get('y') or node_data.get('lat')
            lng_val = node_data.get('x') or node_data.get('lon') or node_data.get('lng')
            if lat_val is not None and lng_val is not None:
                coords_map[str(node_id)] = [float(lat_val), float(lng_val)]
                continue
        
        missing_count += 1
        logger.warning(f"Could not find coordinates for node {node_id}")

    logger.info(f"Mapped {len(coords_map)} / {len(nodes)} nodes. Missing: {missing_count}")

    output_path = models_dir / "m4_nodes_coords.json"
    with open(output_path, "w") as f:
        json.dump(coords_map, f, indent=2)
    
    logger.info(f"Successfully saved coordinates mapping to {output_path}")

if __name__ == "__main__":
    generate()
