"""Shared road graph construction from OpenStreetMap.

Builds a Bengaluru road network graph using osmnx, caches it as GraphML.
Both geo_impute.py and m4_gnn_backbone.py consume this cached graph.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load data config from configs/data.yaml."""
    cfg = OmegaConf.load("configs/data.yaml")
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def get_or_build_road_graph(config: dict | None = None):  # type: ignore[no-untyped-def]
    """Get cached road graph or build from OSM.

    Args:
        config: Optional config dict. Loaded from configs/data.yaml if None.

    Returns:
        A networkx MultiDiGraph of the Bengaluru road network.
    """
    try:
        import networkx as nx
        import osmnx as ox
    except ImportError as e:
        logger.error(
            "osmnx and networkx are required for road graph construction. "
            "Install with: uv sync --extra deep"
        )
        raise ImportError(
            "osmnx/networkx not installed. Run 'uv sync --extra deep' first."
        ) from e

    if config is None:
        config = load_config()

    geo_config = config.get("geo_impute", {})
    cache_path = Path(config["paths"]["osm_graph_cache_path"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Return cached graph if it exists
    if cache_path.exists():
        logger.info("Loading cached road graph from %s", cache_path)
        try:
            if cache_path.suffix == ".gz":
                import gzip
                with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                    graphml_data = f.read()
                    graph = ox.load_graphml(graphml_str=graphml_data)
            else:
                graph = ox.load_graphml(cache_path)
        except Exception as e:
            logger.warning("osmnx load_graphml failed (%s). Falling back to direct networkx read_graphml", e)
            try:
                if cache_path.suffix == ".gz":
                    import gzip
                    with gzip.open(cache_path, "rb") as f:
                        graph = nx.read_graphml(f)
                else:
                    graph = nx.read_graphml(cache_path)
            except Exception as ex:
                logger.error("Failed to load road graph: %s", ex)
                raise ex
        logger.info(
            "Loaded graph: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph

    # Build from OSM
    place = geo_config.get("osm_place", "Bengaluru, India")
    network_type = geo_config.get("osm_network_type", "drive")

    logger.info("Downloading OSM road network for '%s' (type=%s)...", place, network_type)
    graph = ox.graph_from_place(place, network_type=network_type)
    logger.info(
        "Built graph: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    # Save cache
    if cache_path.suffix == ".gz":
        import gzip
        with gzip.open(cache_path, "wb") as f:
            nx.write_graphml(graph, f)
    else:
        ox.save_graphml(graph, cache_path)
    logger.info("Cached road graph to %s", cache_path)

    return graph


if __name__ == "__main__":
    get_or_build_road_graph()
