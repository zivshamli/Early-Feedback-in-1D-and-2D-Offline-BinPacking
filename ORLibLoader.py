import os
import random
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class BinPackingInstance:
    name: str
    capacity: float
    num_items: int
    optimal_bins: int
    items: list



class ORLib1DBinPackingLoader:

    def __init__(self, path):

        self.path = path
        self.instances = []

        self.train_instances = []
        self.validation_instances = []
        self.test_instances = []



    def load(self):

        if os.path.isfile(self.path):

            self.instances.extend(
                self._load_file(self.path)
            )


        elif os.path.isdir(self.path):

            files = sorted([
                os.path.join(self.path, f)
                for f in os.listdir(self.path)
                if f.endswith(".txt")
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



    def _load_file(self, filepath):

        instances = []


        with open(filepath, "r") as f:

            lines = [
                line.strip()
                for line in f.readlines()
                if line.strip()
            ]


        num_instances = int(lines[0])

        index = 1


        for _ in range(num_instances):

            name = lines[index]
            index += 1


            values = list(
                map(float, lines[index].split())
            )

            capacity = values[0]
            num_items = int(values[1])
            optimal_bins = int(values[2])

            index += 1


            items = []

            for _ in range(num_items):

                items.append(
                    float(lines[index])
                )

                index += 1



            instances.append(
                BinPackingInstance(
                    name=name,
                    capacity=capacity,
                    num_items=num_items,
                    optimal_bins=optimal_bins,
                    items=items
                )
            )


        return instances



    # =====================================================
    # New Stratified Split
    # =====================================================

    def train_val_test_split(
            self,
            test_size=0.20,
            val_size=0.15,
            seed=42
    ):

        if len(self.instances) == 0:

            raise ValueError(
                "Load instances first"
            )


        rng = random.Random(seed)


        groups = defaultdict(list)


        # Group examples:
        # t60_00  -> t60
        # u120_05 -> u120

        for inst in self.instances:

            family = inst.name.split("_")[0]

            groups[family].append(inst)



        train = []
        validation = []
        test = []



        for family, instances in groups.items():

            rng.shuffle(instances)


            n = len(instances)


            n_test = max(
                1,
                int(round(n * test_size))
            )


            test_group = instances[:n_test]


            remaining = instances[n_test:]


            n_val = max(
                1,
                int(round(len(remaining) * val_size))
            )


            validation_group = remaining[:n_val]


            train_group = remaining[n_val:]



            train.extend(train_group)

            validation.extend(validation_group)

            test.extend(test_group)



        rng.shuffle(train)
        rng.shuffle(validation)
        rng.shuffle(test)



        self.train_instances = train
        self.validation_instances = validation
        self.test_instances = test



        return train, validation, test




    # =====================================================
    # Old split (kept)
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


        rng = random.Random(seed)


        data = self.instances.copy()

        rng.shuffle(data)


        split = int(
            len(data)*(1-test_size)
        )


        train = data[:split]

        test = data[split:]


        self.train_instances = train

        self.test_instances = test


        return train, test



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



    def get_instance(self,index):

        return self.instances[index]



    def summary(self):

        print(
            f"Total instances: {len(self.instances)}"
        )


        for inst in self.instances[:5]:

            print(
                f"{inst.name}: "
                f"items={inst.num_items}, "
                f"capacity={inst.capacity}, "
                f"opt={inst.optimal_bins}"
            )



    def __len__(self):

        return len(self.instances)



    def __iter__(self):

        return iter(self.instances)





# ===========================
# Test
# ===========================

if __name__ == "__main__":


    loader = ORLib1DBinPackingLoader(
        "ORLIB"
    )


    loader.load()


    train, val, test = loader.train_val_test_split(
        test_size=0.2,
        val_size=0.15,
        seed=42
    )


    print(
        "Train:",
        len(train)
    )

    print(
        "Validation:",
        len(val)
    )

    print(
        "Test:",
        len(test)
    )


    loader.check_split()
