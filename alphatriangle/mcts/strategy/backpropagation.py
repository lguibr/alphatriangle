import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.node import Node

logger = logging.getLogger(__name__)


def backpropagate_value(leaf_node: "Node", value: float) -> int:
    """
    Propagates the simulation value back up the tree from the leaf node.
    Returns the depth of the backpropagation path (number of nodes updated).
    """
    current_node: Node | None = leaf_node
    path_str = []
    depth = 0
    logger.debug(
        f"[Backprop] Starting backprop from leaf (Action={leaf_node.action_taken}, StateStep={leaf_node.state.current_step}) with value={value:.4f}"
    )

    while current_node is not None:
        q_before = current_node.value_estimate
        total_val_before = current_node.total_action_value
        visits_before = current_node.visit_count

        current_node.visit_count += 1
        current_node.total_action_value += value

        q_after = current_node.value_estimate
        total_val_after = current_node.total_action_value
        visits_after = current_node.visit_count

        action_str = (
            f"Act={current_node.action_taken}"
            if current_node.action_taken is not None
            else "Root"
        )
        path_str.append(f"N({action_str},V={visits_after},Q={q_after:.3f})")

        logger.debug(
            f"  [Backprop] Depth {depth}: Node({action_str}), "
            f"Visits: {visits_before} -> {visits_after}, "
            f"AddedVal={value:.4f}, "
            f"TotalVal: {total_val_before:.3f} -> {total_val_after:.3f}, "
            f"Q: {q_before:.3f} -> {q_after:.3f}"
        )

        current_node = current_node.parent
        depth += 1

    logger.debug(f"[Backprop] Finished. Path: {' <- '.join(reversed(path_str))}")
    return depth
