import cupy
from typing import Any
import cudf
from tqdm import tqdm
import pylibcugraph as libcugraph
import numpy as np
from gq.utils.graphs import edge_index_to_cugraph


def compute_apsp(
    dataset, depth_limit=10, flatten_every=10000, dtype=np.byte
) -> np.ndarray:
    data: Any = dataset.data_list[0]
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    g, handle = edge_index_to_cugraph(edge_index)

    result: np.ndarray = np.empty((num_nodes, num_nodes), dtype=dtype)

    if result.dtype != np.int32:
        max_val = np.iinfo(result.dtype).max
    else:
        max_val = None

    for i in tqdm(range(num_nodes), desc="Computing APSP"):
        distances: cupy.ndarray
        distances, _, _ = libcugraph.bfs(
            handle=handle,
            graph=g,
            sources=cudf.Series([i], dtype="int32"),
            direction_optimizing=True,
            depth_limit=depth_limit,
            compute_predecessors=False,
            do_expensive_check=False,
        )
        distances = distances.astype(dtype)
        if max_val is not None:
            distances[distances < 0] = max_val
        cupy.asnumpy(
            distances.astype(dtype),
            out=result[i],
            blocking=True,
        )

    # cupy.cuda.get_current_stream().synchronize()

    return result
