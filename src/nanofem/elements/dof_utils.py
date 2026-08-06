"""Local-to-global DOF map construction, shared across multi-node elements (SDS C-2).

``elements/factory.py`` builds ``Bar``'s trivial 2-node/1-component DOF pair
by hand; any element with more nodes or more components per node needs the
general node-major-then-component array this module provides once, rather
than every element author re-deriving the same double loop.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from nanofem.core.dof_handler import DofHandler


def build_local_dof_map(
    dof_handler: DofHandler,
    node_ids: Sequence[int],
    field: str,
    components: tuple[str, ...],
) -> NDArray[np.int64]:
    """Global DOFs for ``(node, field, component)``, node-major-then-component order.

    Local index ``d = node_index * len(components) + component_index`` (SDS
    C-2) - the same ordering a C-order ``reshape`` gives when flattening an
    operator's trailing ``(n_fun, dim)`` axes, so this map lines up directly
    with a flattened B-matrix's columns.
    """
    return np.array(
        [
            dof_handler.global_dof(node_id, field, component)
            for node_id in node_ids
            for component in components
        ],
        dtype=np.int64,
    )
