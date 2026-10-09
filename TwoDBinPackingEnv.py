
import numpy as np
import matplotlib.pyplot as plt


class Offline2DBinPackingEnv:
    """
    Offline 2D Bin Packing Environment.

    Each item is represented as:
        (width, height)

    Each action is represented as:
        (item_index, bin_index, x, y, rotation)

    where:
        item_index : index of the item in remaining_items
        bin_index  : index of the bin
        x, y       : bottom-left position of the item
        rotation   : 0 or 90 degrees

    A new bin is opened when:
        bin_index == len(self.bins)

    The environment supports:
        - 2D rectangular packing
        - 90-degree item rotation
        - overlap checking
        - boundary checking
        - stability checking
        - utilization
        - packing density based on item contacts
        - item stability metric
        - fixed-size state vector
        - visualization
    """

    def __init__(
        self,
        items,
        bin_width,
        bin_height,
        max_items=100,
        stability_threshold=0.5,
        grid_size=1
    ):
        """
        Parameters
        ----------
        items : list
            List of items represented as (width, height).

        bin_width : int or float
            Width of each bin.

        bin_height : int or float
            Height of each bin.

        max_items : int
            Maximum number of items supported by the state vector.

        stability_threshold : float
            Minimum fraction of the item's width that must be exceeded
            for an elevated item to be considered stable.

        grid_size : int or float
            Spatial discretization used for positions.
            grid_size=1 means integer positions.
        """

        self.items = [
            (float(item[0]), float(item[1]))
            for item in items
        ]

        self.bin_width = float(bin_width)
        self.bin_height = float(bin_height)

        self.max_items = max_items
        self.stability_threshold = stability_threshold
        self.grid_size = grid_size

        self.reset()

    # ============================================================
    # RESET
    # ============================================================

    def reset(self):
        """
        Reset the environment.

        Each bin is represented as a list of dictionaries:

        {
            "item_index": int,
            "x": float,
            "y": float,
            "width": float,
            "height": float,
            "rotation": int
        }
        """

        self.bins = []

        # Same purpose as in the 1D environment
        self.map_index_item_in_bins = []

        self.remaining_items = self.items.copy()

        # Original indices of remaining items
        self.remaining_item_indices = list(range(len(self.items)))

        self.done = False

        return self._get_state()

    # ============================================================
    # STATE
    # ============================================================

    def _get_state(self):
        """
        Returns a dictionary representation of the current state.
        """

        return {
            "bins": self.bins.copy(),
            "remaining_items": self.remaining_items.copy(),
            "remaining_item_indices": self.remaining_item_indices.copy()
        }

    # ============================================================
    # ITEM DIMENSIONS
    # ============================================================

    def _get_rotated_dimensions(self, item, rotation):
        """
        Return width and height after rotation.

        rotation = 0:
            (width, height)

        rotation = 90:
            (height, width)
        """

        width, height = item

        if rotation == 0:
            return width, height

        if rotation == 90:
            return height, width

        raise ValueError(
            "Invalid rotation. Rotation must be 0 or 90."
        )

    # ============================================================
    # VALID ACTIONS
    # ============================================================

    def get_valid_actions(self):
        """
        Returns all valid actions for the current state.

        Action format:

            (item_index, bin_index, x, y, rotation)

        For each remaining item:
            - try rotation 0
            - try rotation 90
            - try every feasible position
            - try every existing bin
            - try opening a new bin

        Positions are generated according to grid_size.
        """

        valid_actions = []

        for item_index, item in enumerate(self.remaining_items):

            for rotation in [0, 90]:

                width, height = self._get_rotated_dimensions(
                    item,
                    rotation
                )

                # Item cannot fit inside an empty bin
                if width > self.bin_width or height > self.bin_height:
                    continue

                # ------------------------------------------------
                # Existing bins
                # ------------------------------------------------

                for bin_index, bin_items in enumerate(self.bins):

                    positions = self._get_candidate_positions(
                        bin_items,
                        width,
                        height
                    )

                    for x, y in positions:

                        if self._can_place(
                            bin_index,
                            x,
                            y,
                            width,
                            height
                        ):

                            valid_actions.append(
                                (
                                    item_index,
                                    bin_index,
                                    x,
                                    y,
                                    rotation
                                )
                            )

                # ------------------------------------------------
                # New bin
                # ------------------------------------------------

                if width <= self.bin_width and height <= self.bin_height:

                    # First item in an empty bin
                    valid_actions.append(
                        (
                            item_index,
                            len(self.bins),
                            0,
                            0,
                            rotation
                        )
                    )

        return valid_actions

    # ============================================================
    # CANDIDATE POSITIONS
    # ============================================================

    def _get_candidate_positions(
        self,
        bin_items,
        width,
        height
    ):
        """
        Generate candidate positions.

        Instead of checking every coordinate in the bin, candidate
        positions are generated from:
            - (0, 0)
            - right side of existing items
            - top side of existing items

        This is much more efficient than scanning the complete
        2D grid.
        """

        positions = set()

        # Always consider bottom-left corner
        positions.add((0.0, 0.0))

        for placed_item in bin_items:

            px = placed_item["x"]
            py = placed_item["y"]
            pw = placed_item["width"]
            ph = placed_item["height"]

            # Right side
            positions.add(
                (
                    px + pw,
                    py
                )
            )

            # Top side
            positions.add(
                (
                    px,
                    py + ph
                )
            )

            # Top-right corner
            positions.add(
                (
                    px + pw,
                    py + ph
                )
            )

        # Remove positions outside the bin
        valid_positions = []

        for x, y in positions:

            if (
                x >= 0
                and y >= 0
                and x + width <= self.bin_width
                and y + height <= self.bin_height
            ):
                valid_positions.append((x, y))

        return valid_positions

    # ============================================================
    # STEP
    # ============================================================

    def step(self, action, global_item_index=None):
        """
        Execute an action.

        Action:
            (item_index, bin_index, x, y, rotation)

        Returns:
            state,
            reward,
            done,
            info
        """

        if self.done:
            raise Exception(
                "Episode has ended. Please reset the environment."
            )

        if len(action) != 5:
            raise ValueError(
                "Action must be "
                "(item_index, bin_index, x, y, rotation)."
            )

        item_index, bin_index, x, y, rotation = action

        # --------------------------------------------------------
        # Validate item index
        # --------------------------------------------------------

        if (
            item_index < 0
            or item_index >= len(self.remaining_items)
        ):
            raise ValueError("Invalid item index.")

        # --------------------------------------------------------
        # Validate rotation
        # --------------------------------------------------------

        if rotation not in [0, 90]:
            raise ValueError(
                "Invalid rotation. Rotation must be 0 or 90."
            )

        item = self.remaining_items[item_index]

        width, height = self._get_rotated_dimensions(
            item,
            rotation
        )

        # --------------------------------------------------------
        # Determine global item index
        # --------------------------------------------------------

        if global_item_index is None:
            global_item_index = self.remaining_item_indices[
                item_index
            ]

        # --------------------------------------------------------
        # New bin
        # --------------------------------------------------------

        if bin_index == len(self.bins):

            if not self._fits_inside_bin(
                x,
                y,
                width,
                height
            ):
                return (
                    self._get_state(),
                    -10,
                    False,
                    {
                        "invalid_action": True,
                        "reason": "item_out_of_bounds"
                    }
                )

            # A new bin starts empty
            if x != 0 or y != 0:

                # For a new bin, force the first item to the
                # bottom-left position.
                return (
                    self._get_state(),
                    -10,
                    False,
                    {
                        "invalid_action": True,
                        "reason":
                            "new_bin_item_must_start_at_0_0"
                    }
                )

            new_item = {
                "item_index": global_item_index,
                "x": float(x),
                "y": float(y),
                "width": float(width),
                "height": float(height),
                "rotation": rotation
            }

            self.bins.append([new_item])

            self.map_index_item_in_bins.append(
                [global_item_index]
            )

        # --------------------------------------------------------
        # Existing bin
        # --------------------------------------------------------

        elif 0 <= bin_index < len(self.bins):

            if not self._can_place(
                bin_index,
                x,
                y,
                width,
                height
            ):

                return (
                    self._get_state(),
                    -10,
                    False,
                    {
                        "invalid_action": True,
                        "reason": "invalid_placement"
                    }
                )

            new_item = {
                "item_index": global_item_index,
                "x": float(x),
                "y": float(y),
                "width": float(width),
                "height": float(height),
                "rotation": rotation
            }

            self.bins[bin_index].append(new_item)

            self.map_index_item_in_bins[
                bin_index
            ].append(global_item_index)

        else:

            raise ValueError("Invalid bin index.")

        # --------------------------------------------------------
        # Remove item from remaining items
        # --------------------------------------------------------

        self.remaining_items.pop(item_index)

        self.remaining_item_indices.pop(item_index)

        # --------------------------------------------------------
        # Check termination
        # --------------------------------------------------------

        if not self.remaining_items:
            self.done = True

        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------

        utilization = self._calculate_utilization()
        density = self._calculate_density()
        stability = self._calculate_stability()

        info = {
            "bins_used": len(self.bins),
            "order_bins": self.bins,
            "item_indices_in_bins":
                self.map_index_item_in_bins,
            "utilization": utilization,
            "packing_density": density,
            "stability": stability,
            "invalid_action": False
        }

        # The environment itself does not implement Early Feedback.
        # The reward here is kept compatible with the 1D environment:
        # opening a new bin receives -1.
        reward = 0

        if bin_index == len(self.bins) - 1:
            # This condition is not sufficient to identify whether
            # a new bin was actually opened after append.
            # Therefore reward is recalculated below.
            pass

        # New bin was opened if the placed item is the first item
        # in the last bin.
        if (
            len(self.bins[-1]) == 1
            and self.map_index_item_in_bins[-1][0]
            == global_item_index
        ):
            reward = -1

        return (
            self._get_state(),
            reward,
            self.done,
            info
        )

    # ============================================================
    # VALID PLACEMENT
    # ============================================================

    def _can_place(
        self,
        bin_index,
        x,
        y,
        width,
        height
    ):
        """
        Check whether an item can be placed in a bin.
        """

        # Check bin index
        if (
            bin_index < 0
            or bin_index >= len(self.bins)
        ):
            return False

        # Check boundaries
        if not self._fits_inside_bin(
            x,
            y,
            width,
            height
        ):
            return False

        # Check overlap
        if self._check_overlap(
            bin_index,
            x,
            y,
            width,
            height
        ):
            return False

        # Check stability
        if not self._check_stability(
            bin_index,
            x,
            y,
            width,
            height
        ):
            return False

        return True

    # ============================================================
    # BOUNDARY CHECK
    # ============================================================

    def _fits_inside_bin(
        self,
        x,
        y,
        width,
        height
    ):
        """
        Check whether the rectangle stays inside the bin.
        """

        return (
            x >= 0
            and y >= 0
            and x + width <= self.bin_width
            and y + height <= self.bin_height
        )

    # ============================================================
    # OVERLAP
    # ============================================================

    def _check_overlap(
        self,
        bin_index,
        x,
        y,
        width,
        height
    ):
        """
        Return True if the proposed rectangle overlaps an
        existing item.

        Touching boundaries is allowed.
        """

        new_left = x
        new_right = x + width
        new_bottom = y
        new_top = y + height

        for placed_item in self.bins[bin_index]:

            old_left = placed_item["x"]
            old_right = (
                placed_item["x"]
                + placed_item["width"]
            )

            old_bottom = placed_item["y"]
            old_top = (
                placed_item["y"]
                + placed_item["height"]
            )

            horizontal_overlap = (
                new_left < old_right
                and new_right > old_left
            )

            vertical_overlap = (
                new_bottom < old_top
                and new_top > old_bottom
            )

            if (
                horizontal_overlap
                and vertical_overlap
            ):
                return True

        return False

    # ============================================================
    # STABILITY
    # ============================================================

    def _check_stability(
        self,
        bin_index,
        x,
        y,
        width,
        height
    ):
        """
        Check whether an item is stable.

        Definition:

        1. Item placed on the floor:
               y == 0

        OR

        2. Item is sufficiently supported by items below it.

        The support ratio is:

            supported_width / item_width

        and must be >= stability_threshold.
        """

        # Item on the floor
        if np.isclose(y, 0.0):
            return True

        item_left = x
        item_right = x + width
        item_bottom = y

        supported_segments = []

        for placed_item in self.bins[bin_index]:

            placed_top = (
                placed_item["y"]
                + placed_item["height"]
            )

            # Item must be directly underneath
            if not np.isclose(
                placed_top,
                item_bottom
            ):
                continue

            placed_left = placed_item["x"]

            placed_right = (
                placed_item["x"]
                + placed_item["width"]
            )

            overlap_left = max(
                item_left,
                placed_left
            )

            overlap_right = min(
                item_right,
                placed_right
            )

            overlap_width = (
                overlap_right - overlap_left
            )

            if overlap_width > 0:
                supported_segments.append(
                    (
                        overlap_left,
                        overlap_right
                    )
                )

        if not supported_segments:
            return False

        # Merge support intervals
        supported_segments.sort()

        merged = [
            supported_segments[0]
        ]

        for start, end in supported_segments[1:]:

            last_start, last_end = merged[-1]

            if start <= last_end:
                merged[-1] = (
                    last_start,
                    max(last_end, end)
                )
            else:
                merged.append(
                    (start, end)
                )

        supported_width = sum(
            end - start
            for start, end in merged
        )

        support_ratio = (
            supported_width / width
            if width > 0
            else 0
        )

        return (
            support_ratio
            >= self.stability_threshold
        )

    # ============================================================
    # UTILIZATION
    # ============================================================

    def _calculate_utilization(self):
        """
        Area utilization:

            total item area
            ------------------------------
            number_of_bins * bin_area
        """

        if len(self.bins) == 0:
            return 0.0

        total_capacity = (
            len(self.bins)
            * self.bin_width
            * self.bin_height
        )

        total_used = 0.0

        for bin_items in self.bins:

            for item in bin_items:

                total_used += (
                    item["width"]
                    * item["height"]
                )

        return (
            total_used / total_capacity
            if total_capacity > 0
            else 0.0
        )

    # ============================================================
    # PACKING DENSITY
    # ============================================================

    def _calculate_density(self):
        """
        Calculate Packing Density based on item contacts.

        For each bin, a contact is counted when two items share a
        positive-length boundary segment (horizontal or vertical).
        Corner-only contact is not counted.

        Packing density is normalized to [0, 1]:

            total contacts
            -----------------------------
            total possible pairs per bin
        """

        if len(self.bins) == 0:
            return 0.0

        total_contacts = 0
        total_possible_pairs = 0

        for bin_items in self.bins:
            n = len(bin_items)

            if n < 2:
                continue

            total_possible_pairs += n * (n - 1) // 2

            for i in range(n):
                item_a = bin_items[i]

                a_left = item_a["x"]
                a_right = item_a["x"] + item_a["width"]
                a_bottom = item_a["y"]
                a_top = item_a["y"] + item_a["height"]

                for j in range(i + 1, n):
                    item_b = bin_items[j]

                    b_left = item_b["x"]
                    b_right = item_b["x"] + item_b["width"]
                    b_bottom = item_b["y"]
                    b_top = item_b["y"] + item_b["height"]

                    # Vertical shared boundary:
                    # one item's right side touches the other's left side.
                    vertical_contact = (
                        (
                            np.isclose(a_right, b_left)
                            or np.isclose(b_right, a_left)
                        )
                        and min(a_top, b_top) > max(a_bottom, b_bottom)
                    )

                    # Horizontal shared boundary:
                    # one item's top side touches the other's bottom side.
                    horizontal_contact = (
                        (
                            np.isclose(a_top, b_bottom)
                            or np.isclose(b_top, a_bottom)
                        )
                        and min(a_right, b_right) > max(a_left, b_left)
                    )

                    if vertical_contact or horizontal_contact:
                        total_contacts += 1

        if total_possible_pairs == 0:
            return 0.0

        return total_contacts / total_possible_pairs

    # ============================================================
    # STABILITY METRIC
    # ============================================================

    def _calculate_stability(self):
        """
        Calculate the proportion of stable items.

        An item is stable if:
        1. It is placed directly on the bin floor, or
        2. More than stability_threshold (default 50%) of its base
           is supported by items directly below it.

        Therefore:

            ItemStability =
                number of stable items / total number of items

        The result is in [0, 1].
        """

        total_items = 0
        stable_items = 0

        for bin_items in self.bins:
            for item in bin_items:
                total_items += 1

                x = item["x"]
                y = item["y"]
                width = item["width"]

                # Items placed on the floor are stable.
                if np.isclose(y, 0.0):
                    stable_items += 1
                    continue

                item_left = x
                item_right = x + width
                item_bottom = y

                supported_segments = []

                for other_item in bin_items:
                    if other_item is item:
                        continue

                    other_top = (
                        other_item["y"] + other_item["height"]
                    )

                    # Support must come directly from below.
                    if not np.isclose(other_top, item_bottom):
                        continue

                    other_left = other_item["x"]
                    other_right = (
                        other_item["x"] + other_item["width"]
                    )

                    overlap_left = max(item_left, other_left)
                    overlap_right = min(item_right, other_right)

                    overlap_width = (
                        overlap_right - overlap_left
                    )

                    if overlap_width > 0:
                        supported_segments.append(
                            (overlap_left, overlap_right)
                        )

                if not supported_segments:
                    continue

                # Merge overlapping support intervals.
                supported_segments.sort()

                merged = [supported_segments[0]]

                for start, end in supported_segments[1:]:
                    last_start, last_end = merged[-1]

                    if start <= last_end:
                        merged[-1] = (
                            last_start,
                            max(last_end, end)
                        )
                    else:
                        merged.append((start, end))

                supported_width = sum(
                    end - start
                    for start, end in merged
                )

                support_ratio = (
                    supported_width / width
                    if width > 0
                    else 0.0
                )

                # Stable only when MORE than 50% is supported.
                if support_ratio > self.stability_threshold:
                    stable_items += 1

        if total_items == 0:
            return 0.0

        return stable_items / total_items

    # ============================================================
    # STATE VECTOR
    # ============================================================

    def get_state_vector(self):
        """
        Return a fixed-size vector representation.

        The vector contains:

        1. Remaining items:
               width, height

        2. Information about existing bins.

        For each bin:
            - used area
            - remaining area
            - number of placed items

        3. Placement information for placed items:
            - x
            - y
            - width
            - height
            - rotation

        The vector is padded with zeros to a fixed size.
        """

        vector = []

        # --------------------------------------------------------
        # Remaining items
        # --------------------------------------------------------

        max_item_values = self.max_items * 2

        for item in self.remaining_items[:self.max_items]:

            vector.append(item[0])
            vector.append(item[1])

        while len(vector) < max_item_values:
            vector.append(0.0)

        # --------------------------------------------------------
        # Existing bins
        # --------------------------------------------------------

        max_bins = self.max_items

        bin_feature_size = 3

        for bin_items in self.bins[:max_bins]:

            used_area = sum(
                item["width"] * item["height"]
                for item in bin_items
            )

            bin_area = (
                self.bin_width
                * self.bin_height
            )

            remaining_area = (
                bin_area - used_area
            )

            number_of_items = len(bin_items)

            vector.extend(
                [
                    used_area,
                    remaining_area,
                    number_of_items
                ]
            )

        while len(vector) < (
            max_item_values
            + max_bins * bin_feature_size
        ):
            vector.append(0.0)

        # --------------------------------------------------------
        # Placed item information
        # --------------------------------------------------------

        max_placed_items = self.max_items

        placed_items = []

        for bin_index, bin_items in enumerate(self.bins):

            for item in bin_items:

                placed_items.append(
                    [
                        float(bin_index),
                        item["x"],
                        item["y"],
                        item["width"],
                        item["height"],
                        float(item["rotation"])
                    ]
                )

        for item in placed_items[:max_placed_items]:

            vector.extend(item)

        max_placement_values = (
            max_placed_items * 6
        )

        current_placement_values = (
            min(
                len(placed_items),
                max_placed_items
            )
            * 6
        )

        while current_placement_values < max_placement_values:

            vector.append(0.0)
            current_placement_values += 1

        return np.array(
            vector,
            dtype=np.float32
        )

    # ============================================================
    # RENDER
    # ============================================================

    def render(self, figsize=(8, 6)):
        """
        Visualize the current packing configuration.

        Each bin is displayed separately.
        """

        number_of_bins = len(self.bins)

        if number_of_bins == 0:

            fig, ax = plt.subplots(
                figsize=figsize
            )

            ax.set_xlim(
                0,
                self.bin_width
            )

            ax.set_ylim(
                0,
                self.bin_height
            )

            ax.set_title(
                "2D Bin Packing - Empty"
            )

            ax.set_xlabel("Width")
            ax.set_ylabel("Height")

            plt.show()

            return

        fig, axes = plt.subplots(
            number_of_bins,
            1,
            figsize=(
                figsize[0],
                figsize[1] * number_of_bins
            )
        )

        if number_of_bins == 1:
            axes = [axes]

        for bin_index, ax in enumerate(axes):

            ax.set_xlim(
                0,
                self.bin_width
            )

            ax.set_ylim(
                0,
                self.bin_height
            )

            ax.set_aspect("equal")

            bin_items = self.bins[bin_index]

            for item in bin_items:

                rectangle = plt.Rectangle(
                    (
                        item["x"],
                        item["y"]
                    ),
                    item["width"],
                    item["height"],
                    fill=False,
                    linewidth=2
                )

                ax.add_patch(rectangle)

                ax.text(
                    item["x"]
                    + item["width"] / 2,
                    item["y"]
                    + item["height"] / 2,
                    str(item["item_index"]),
                    ha="center",
                    va="center"
                )

            ax.set_title(
                f"Bin {bin_index}"
            )

            ax.set_xlabel("Width")
            ax.set_ylabel("Height")

            ax.grid(True)

        plt.tight_layout()
        plt.show()

    # ============================================================
    # PUBLIC METRIC METHODS
    # ============================================================

    def calculate_metrics(self):
        """
        Return all current packing metrics.
        """

        return {
            "bins_used": len(self.bins),
            "utilization":
                self._calculate_utilization(),
            "packing_density":
                self._calculate_density(),
            "stability":
                self._calculate_stability()
        }