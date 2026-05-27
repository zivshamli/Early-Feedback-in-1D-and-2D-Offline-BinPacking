import os
from dataclasses import dataclass

@dataclass
class BinPackingInstance:
    name: str
    capacity: int
    num_items: int
    optimal_bins: int
    items: list

class ORLib1DBinPackingLoader:
    def __init__(self,pathfile):
        self.pathfile=pathfile
        self.instances = []
    
    def load(self):
        
        with open(self.pathfile, "r") as f:

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
            capacity, num_items, optimal_bins = map(int, lines[index].split())
            index += 1
            items = []

            for _ in range(num_items):

                items.append(int(lines[index]))
                index += 1

            instance = BinPackingInstance(
                name=name,
                capacity=capacity,
                num_items=num_items,
                optimal_bins=optimal_bins,
                items=items
            )
            self.instances.append(instance)

        return self.instances   

    def get_instance(self,index):
        if index < 0 or index >= len(self.instances):
            raise IndexError("Instance index out of range.")
        return self.instances[index]
    
    def get_instance_by_name(self,name):
        for instance in self.instances:
            if instance.name == name:
                return instance
        raise ValueError(f"Instance with name '{name}' not found.")
    
    def summary(self):
        
        print(f"Loaded {len(self.instances)} instances:")

        for instance in self.instances[:5]:  # Print details of the first 5 instances
            print(f"Name: {instance.name}, Capacity: {instance.capacity}, Num Items: {instance.num_items}, Optimal Bins: {instance.optimal_bins}")
    def __len__(self):
        return len(self.instances)

    def __iter__(self):
        return iter(self.instances)
    

if __name__ == "__main__":
    loader = ORLib1DBinPackingLoader("binpack1.txt")
    instances = loader.load()
    loader.summary()    
