
import os
import json
import random

from dataclasses import dataclass
from collections import defaultdict


@dataclass
class BinPacking2DInstance:
    name: str
    bin_width: float
    bin_height: float
    num_items: int
    optimal_bins: int
    lower_bound: int
    items: list


class PackLib2DLoader:

    def __init__(self, path):

        self.path = path
        self.instances = []

        self.train_instances = []
        self.validation_instances = []
        self.test_instances = []

    # =====================================================
    # Load
    # =====================================================

    def load(self):

        if os.path.isfile(self.path):

            self.instances.extend(
                self._load_file(self.path)
            )

        elif os.path.isdir(self.path):

            files = sorted([
                os.path.join(self.path, f)
                for f in os.listdir(self.path)
                if f.lower().endswith(".json")
            ])

            for file in files:

                self.instances.extend(
                    self._load_file(file)
                )

        else:

            raise FileNotFoundError(
                f"Path {self.path} does not exist"
            )

        return self.instances

    # =====================================================
    # Load JSON file
    # =====================================================

    def _load_file(self, filepath):

        instances = []

        with open(filepath, "r", encoding="utf-8") as f:

            data = json.load(f)

        name = data["Name"]

        # Bin dimensions
        bin_data = data["Objects"][0]

        bin_width = float(bin_data["Length"])
        bin_height = float(bin_data["Height"])

        # Expand items according to Demand
        items = []

        for item in data["Items"]:

            width = float(item["Length"])
            height = float(item["Height"])

            demand = int(item.get("Demand", 1))

            if demand < 0:
                raise ValueError(
                    f"Negative Demand in instance {name}"
                )

            for _ in range(demand):

                items.append((width, height))

        num_items = len(items)

        # Solution information
        solution = data.get("Solution", {})

        optimal_bins = solution.get("OptimalBins")
        lower_bound = solution.get("LowerBound")

        if optimal_bins is not None:
            optimal_bins = int(optimal_bins)

        if lower_bound is not None:
            lower_bound = int(lower_bound)

        instances.append(
            BinPacking2DInstance(
                name=name,
                bin_width=bin_width,
                bin_height=bin_height,
                num_items=num_items,
                optimal_bins=optimal_bins,
                lower_bound=lower_bound,
                items=items
            )
        )

        return instances

    # =====================================================
    # Stratified Train / Validation / Test Split
    # =====================================================


    def train_val_test_split(
        self,
        test_size=0.20,
        val_size=0.10,
        seed=42
    ):

        if len(self.instances) == 0:
            raise ValueError("Load instances first")

        if not 0 <= test_size < 1:
            raise ValueError(
                "test_size must be in [0, 1)"
            )

        if not 0 <= val_size < 1:
            raise ValueError(
                "val_size must be in [0, 1)"
            )

        if test_size + val_size >= 1:
            raise ValueError(
                "test_size + val_size must be less than 1"
            )

        rng = random.Random(seed)

        data = self.instances.copy()
        rng.shuffle(data)

        n = len(data)

        n_test = round(n * test_size)
        n_val = round(n * val_size)

        test = data[:n_test]

        validation = data[n_test:n_test + n_val]

        train = data[n_test + n_val:]

        self.train_instances = train
        self.validation_instances = validation
        self.test_instances = test

        return train, validation, test



    # =====================================================
    # Train / Test Split
    # =====================================================

    def train_test_split(
        self,
        test_size=0.2,
        seed=42
    ):

        if len(self.instances) == 0:

            raise ValueError(
                "Load instances first"
            )

        if not 0 <= test_size < 1:

            raise ValueError(
                "test_size must be in [0, 1)"
            )

        rng = random.Random(seed)

        data = self.instances.copy()
        rng.shuffle(data)

        split = int(
            len(data) * (1 - test_size)
        )

        train = data[:split]
        test = data[split:]

        self.train_instances = train
        self.validation_instances = []
        self.test_instances = test

        return train, test

    # =====================================================
    # Check Split
    # =====================================================

    def check_split(self):

        train_names = {
            x.name for x in self.train_instances
        }

        val_names = {
            x.name for x in self.validation_instances
        }

        test_names = {
            x.name for x in self.test_instances
        }

        print(
            "Train ∩ Validation:",
            train_names & val_names
        )

        print(
            "Train ∩ Test:",
            train_names & test_names
        )

        print(
            "Validation ∩ Test:",
            val_names & test_names
        )

    # =====================================================
    # Get Instance
    # =====================================================

    def get_instance(self, index):

        return self.instances[index]

    # =====================================================
    # Summary
    # =====================================================

    def summary(self):

        print(
            f"Total instances: {len(self.instances)}"
        )

        for inst in self.instances[:5]:

            print(
                f"{inst.name}: "
                f"items={inst.num_items}, "
                f"bin={inst.bin_width}x{inst.bin_height}, "
                f"opt={inst.optimal_bins}, "
                f"LB={inst.lower_bound}"
            )

    # =====================================================
    # Length / Iterator
    # =====================================================

    def __len__(self):

        return len(self.instances)

    def __iter__(self):

        return iter(self.instances)


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    loader = PackLib2DLoader(
        "packlib2"
    )

    loader.load()

    loader.summary()

    train, val, test = loader.train_val_test_split(
        test_size=0.2,
        val_size=0.1,
        seed=42
    )

    print("Train:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))

    loader.check_split()

    # Example: access first instance
    if len(loader) > 0:

        instance = loader.get_instance(0)

        print("First instance:", instance.name)
        print("Bin dimensions:",
              instance.bin_width,
              instance.bin_height)

        print("Number of items:", instance.num_items)
        print("Items:", instance.items)
        print("Optimal bins:", instance.optimal_bins)
        print("Lower bound:", instance.lower_bound)

