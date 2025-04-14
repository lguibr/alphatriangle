import numpy as np

from .types import Experience


class SumTree:
    """
    Simple SumTree implementation for efficient prioritized sampling.
    Stores priorities and allows sampling proportional to priority.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity

        # Tree structure: Stores priorities. Size is 2*capacity - 1.
        # Leaves are indices capacity-1 to 2*capacity-2.
        self.tree = np.zeros(2 * capacity - 1)

        # Data storage: Stores the actual experiences. Size is capacity.
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0  # Points to the next available data slot
        self.n_entries = 0  # Current number of entries in the buffer
        self._max_priority = 1.0  # Track max priority for new entries

    def add(self, priority: float, data: Experience):
        """Adds an experience with a given priority."""
        # Calculate the tree index for the leaf corresponding to the data slot
        tree_idx = self.data_pointer + self.capacity - 1

        # Store the data
        self.data[self.data_pointer] = data

        # Update the tree with the new priority
        self.update(tree_idx, priority)

        # Move data pointer
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0  # Wrap around

        # Update entry count
        if self.n_entries < self.capacity:
            self.n_entries += 1

        # Update max priority seen
        self._max_priority = max(self._max_priority, priority)

    def update(self, tree_idx: int, priority: float):
        """Updates the priority of an experience at a given tree index."""
        # Calculate the change in priority
        change = priority - self.tree[tree_idx]

        # Update the leaf node
        self.tree[tree_idx] = priority

        # Propagate the change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2  # Move to parent index
            self.tree[tree_idx] += change

    def get_leaf(self, value: float) -> tuple[int, float, Experience]:
        """Finds the leaf node corresponding to a given value (for sampling)."""
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            # If left child index is out of bounds, we've reached a leaf node
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # If the value is less than or equal to the left child's priority sum,
                # go down the left branch.
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                # Otherwise, subtract the left child's sum and go down the right branch.
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        # Calculate the corresponding data index in the self.data array
        data_idx = leaf_idx - self.capacity + 1

        # Return the tree index, the priority at that leaf, and the data
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self) -> float:
        """Returns the total priority (root node value)."""
        # Ensure return type is float
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        """Returns the maximum priority seen so far."""
        # Return 1.0 if buffer is empty to avoid issues with initial adds
        return self._max_priority if self.n_entries > 0 else 1.0

    def __len__(self) -> int:
        return self.n_entries
