import numpy as np


class BinPackingMetrics:

    def __init__(self, bin_capacity, optimal_bins=None):

        self.C = bin_capacity
        self.optimal_bins = optimal_bins

    # ---------------------------
    # 1. Number of bins used
    # ---------------------------
    def num_bins_used(self, solution_bins):

        return len(solution_bins)

    # ---------------------------
    # 2. Utilization
    # ---------------------------
    def utilization(self, solution_bins):

        total_used = sum(sum(b) for b in solution_bins)
        total_capacity = len(solution_bins) * self.C

        return total_used / total_capacity if total_capacity > 0 else 0.0

    # ---------------------------
    # 3. Optimality Gap
    # ---------------------------
    def optimality_gap(self, solution_bins):

        if self.optimal_bins is None:
            raise ValueError("Optimal bins not provided")

        model_bins = len(solution_bins)

        return (model_bins - self.optimal_bins) / self.optimal_bins

    # ---------------------------
    # 4. Convert solution to vector form (for diversity)
    # ---------------------------
    def _solution_to_assignment(self, map_solution_bins, n_items):
        assignment = np.full(n_items, -1, dtype=int)

        seen = set()

        for bin_id, bin_items in enumerate(map_solution_bins):
            for item_index in bin_items:

                if item_index < 0 or item_index >= n_items:
                    raise ValueError(f"Invalid item index: {item_index}")

                if item_index in seen:
                    raise ValueError(f"Duplicate item assignment detected: {item_index}")

                seen.add(item_index)
                assignment[item_index] = bin_id

        # optional: sanity check
        if len(seen) != n_items:
            missing = set(range(n_items)) - seen
            raise ValueError(f"Missing items in solution: {missing}")

        return assignment

    # ---------------------------
    # 5. Hamming distance
    # ---------------------------
    def _hamming(self, s1, s2):

        return np.sum(s1 != s2)

    # ---------------------------
    # 6. Solution Diversity
    # ---------------------------
    def solution_diversity(self, map_solutions, n_items):

        M = len(map_solutions)

        if M < 2:
            return 0.0

        assignments = [
            self._solution_to_assignment(sol, n_items)
            for sol in map_solutions
        ]

        total = 0

        for i in range(M):
            for j in range(i + 1, M):

                total += self._hamming(assignments[i], assignments[j])

        diversity = (2 * total) / (M * (M - 1) * n_items)

        return diversity

    # ---------------------------
    # 7. Stability (over multiple runs)
    # ---------------------------
    def stability(self, results):

        """
        results = list of metrics (e.g., bins used across seeds)
        """

        return np.std(results)

    # ---------------------------
    # 8. Evaluate full episode
    # ---------------------------
    def evaluate_episode(self, solution_bins, n_items):

        return {
            "bins_used": self.num_bins_used(solution_bins),
            "utilization": self.utilization(solution_bins),
            "optimality_gap": (
                self.optimality_gap(solution_bins)
                if self.optimal_bins is not None else None
            ),
            "n_items": n_items
        }