import json
from attr import attrib
import cupy
from networkx import nodes
import numpy as np
from torch import Tensor
from torch_geometric.data import Data
import pylibcugraph as libcugraph

import torch
import torch_geometric.utils as tu
from torch_scatter import scatter_add
import networkx as nx
import igraph as ig
import igraph.remote.gephi as gephi


def degree(edge_index: Tensor, direction="out", num_nodes=None, edge_weight=None):
    """calulcates the degree of each node in the graph

    Args:
        edge_index (Tensor): tensor edge_index encoding the graph structure
        direction (str, optional): either calculate 'in'-degree or 'out'-degree. Defaults to 'out'.
        num_nodes (int, optional): number of nodes. Defaults to None.
        edge_weight (Tensor, optional): weight of edges. Defaults to None.

    Raises:
        AssertionError: raised if unsupported direction is passed

    Returns:
        Tensor: node degree
    """
    row, col = edge_index[0], edge_index[1]

    if num_nodes is None:
        num_nodes = edge_index.max() + 1

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)

    if direction == "out":
        return scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)  # type: ignore

    elif direction == "in":
        return scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)  # type: ignore

    else:
        raise AssertionError


def get_k_hop_diversity(data: Data, k=1, kind="diversity"):
    """returns k-hop-diversity of each node in the grap

    Args:
        data (Data): pytorch-geometric data object representing graph
        k (int, optional): k specifying k-hop neighborhood. Defaults to 1.
        kind (str, optional): either return 'purity' or 'diversity'. Defaults to 'diversity'.

    Raises:
        AssertionError: raised if unsurported kind is passed

    Returns:
        Tensor: divsierty of purity
    """
    n_nodes = data.y.size(0)  # type: ignore
    diversity = torch.zeros_like(data.y)  # type: ignore

    if kind == "purity":
        diversity = diversity.float()

    for n in range(n_nodes):
        k_hop_nodes, _, _, _ = tu.k_hop_subgraph(n, k, data.edge_index)  # type: ignore
        if kind == "diversity":
            div = len(data.y[k_hop_nodes].unique())  # type: ignore
        elif kind == "purity":
            y_center = data.y[n]  # type: ignore
            y_hop = data.y[k_hop_nodes]  # type: ignore
            div = (y_hop == y_center.item()).float().mean()

        else:
            raise AssertionError

        diversity[n] = div

    return diversity


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.ndarray, torch.Tensor)):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        return super().default(o)


def _data_to_graph(data: Data):
    edges = data.edge_index.T.numpy()  # type: ignore

    y = data.y.numpy()  # type: ignore

    nodes = {
        "true_y": y,
        "train_mask": data.train_mask.numpy().astype(int),
        "val_mask": data.val_mask.numpy().astype(int),
        "test_mask": data.test_mask.numpy().astype(int),
    }

    return len(y), nodes, edges


def data_to_nx(data: Data) -> nx.Graph:
    n, nodes, edges = _data_to_graph(data)

    g = nx.Graph()

    g.add_nodes_from(range(n))

    for i in range(n):
        node = g.nodes[i]
        for k, v in nodes.items():
            if v is not None:
                node[k] = v[i]

    g.add_edges_from(edges)

    return g


def data_to_igraph(data: Data) -> ig.Graph:
    n, nodes, edges = _data_to_graph(data)

    g = ig.Graph()

    g.add_vertices(range(n))

    for i in range(n):
        node = g.vs[i]
        for k, v in nodes.items():
            if v is not None:
                node[k] = v[i]

    g.add_edges(edges)

    return g


def edge_index_to_cugraph(edge_index: Tensor | cupy.ndarray) -> libcugraph.SGGraph:
    if isinstance(edge_index, Tensor):
        edge_index = cupy.asarray(edge_index.to(torch.int32))

    handle = libcugraph.ResourceHandle()
    g = libcugraph.SGGraph(
        handle,
        libcugraph.GraphProperties(is_symmetric=True, is_multigraph=False),
        edge_index[0],
        edge_index[1],
        store_transposed=False,
        renumber=False,
        do_expensive_check=False,
    )
    return g, handle


class GephiGraphStreamer(gephi.GephiGraphStreamer):
    def iterjsonobj(self, graph, custom_prefix: str | None = None, delete=False):
        """Iterates over the JSON objects that build up the graph using the
        Gephi graph streaming API. The objects returned from this function are
        Python objects; they must be formatted with ``json.dumps`` before
        sending them to the destination.
        """

        # Construct a unique ID prefix
        if custom_prefix is None:
            id_prefix = "igraph:%s" % (hex(id(graph)),)
            node_prefix = "%s:v:" % (id_prefix,)
            edge_prefix = "%s:e:" % (id_prefix,)
        else:
            node_prefix = ""
            edge_prefix = ""

        if delete:
            # Delete the vertices
            delete_node = self.format.get_delete_node_event
            for vertex in graph.vs:
                yield delete_node("%s%d" % (node_prefix, vertex.index))
        else:
            # Add the vertices
            add_node = self.format.get_add_node_event
            for vertex in graph.vs:
                yield add_node(
                    "%s%d" % (node_prefix, vertex.index), vertex.attributes()
                )
            # Add the edges
            add_edge = self.format.get_add_edge_event
            directed = graph.is_directed()
            for edge in graph.es:
                yield add_edge(
                    "%s%d:%d" % (edge_prefix, edge.source, edge.target),
                    "%s%d" % (node_prefix, edge.source),
                    "%s%d" % (node_prefix, edge.target),
                    directed,
                    edge.attributes(),
                )

    def post(
        self, graph, destination, encoder=None, custom_prefix=None, override=False
    ):
        """Posts the given graph to the destination of the streamer using the
        given JSON encoder. When ``encoder`` is ``None``, it falls back to the
        default JSON encoder of the streamer in the `encoder` property.

        ``destination`` must be a file-like object or an instance of
        `GephiConnection`.
        """
        encoder = encoder or self.encoder

        if override:
            for jsonobj in self.iterjsonobj(
                graph, custom_prefix=custom_prefix, delete=True
            ):
                self.send_event(jsonobj, destination, encoder=encoder, flush=False)
            destination.flush()

        for jsonobj in self.iterjsonobj(graph, custom_prefix=custom_prefix):
            self.send_event(jsonobj, destination, encoder=encoder, flush=False)
        destination.flush()


def vizualize_in_gephi(
    graph: ig.Graph | nx.Graph,
    host="localhost",
    port=8080,
    workspace="streaming",
    override=False,
):
    conn = gephi.GephiConnection(url=f"http://{host}:{port}/{workspace}")
    streamer = GephiGraphStreamer(encoder=NumpyEncoder())
    if isinstance(graph, nx.Graph):
        graph = ig.Graph.from_networkx(graph)
    streamer.post(graph, conn, custom_prefix="", override=override)


def update_vertices_in_gephi(
    attributes: dict[str, np.ndarray] | nx.Graph | ig.Graph,
    host="localhost",
    port=8080,
    workspace="streaming",
):
    conn = gephi.GephiConnection(url=f"http://{host}:{port}/{workspace}")
    streamer = gephi.GephiGraphStreamer(encoder=NumpyEncoder())
    formatter = gephi.GephiGraphStreamingAPIFormat()

    if isinstance(attributes, (nx.Graph, ig.Graph)):
        if isinstance(attributes, nx.Graph):
            attrs = [(i, a) for i, a in attributes.nodes(data=True)]
        else:
            attrs = [(v.index, v.attributes()) for v in attributes.vs]

        for i, a in attrs:
            streamer.send_event(
                formatter.get_change_node_event(str(i), a),
                conn,
                flush=False,
            )
    elif isinstance(attributes, dict):
        v = next(iter(attributes.values()))
        n = len(v)

        for i in range(n):
            streamer.send_event(
                formatter.get_change_node_event(
                    str(i), {k: v[i] for k, v in attributes.items()}
                ),
                conn,
                flush=False,
            )

    conn.flush()
