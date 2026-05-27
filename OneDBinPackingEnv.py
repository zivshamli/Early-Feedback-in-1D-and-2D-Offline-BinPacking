import numpy as np

class Offline1DBinPackingEnv:
    def __init__(self, items,bin_capacity):
        self.items = items
        self.bin_capacity = bin_capacity
        self.reset()

    def reset(self):
        self.bins = [] 
        self.map_index_item_in_bins = []  
        self.remaining_items = self.items.copy()
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        return {
            'bins': self.bins.copy(),
            'remaining_items': self.remaining_items.copy()
        }
    def get_valid_actions(self):

        """
        Returns a list of valid actions (bin indices) for the current state.
        An action is valid if the item can fit into the corresponding bin.
        
        Returns:
            return all valid actions:
            (item_index, bin_index) pairs where the item can be placed in the bin.

            bin index == len(self.bins) means placing the item in a new bin.
        """
        valid_actions=[]

        for item_index, item in enumerate(self.remaining_items):

            #existing bins
            for bin_index, bin in enumerate(self.bins):
                if sum(bin) + item <= self.bin_capacity:
                    valid_actions.append((item_index, bin_index))
            # open a new bin
            if item <= self.bin_capacity:
                valid_actions.append((item_index, len(self.bins)))

        return valid_actions 

    def step(self, action, global_item_index=0):

        """
        Takes an action (item_index, bin_index) and updates the environment state.

        """
        if self.done:
            raise Exception("Episode has ended. Please reset the environment.")
        
        item_index, bin_index = action

        if item_index >= len(self.remaining_items):
            raise ValueError("Invalid item index.")

        item = self.remaining_items[item_index]

        reward = 0

        if bin_index < len(self.bins):
            # Place item in an existing bin
            if sum(self.bins[bin_index]) + item > self.bin_capacity:
                reward = -10
                return self._get_state(), reward, False, {
                    "invalid_action": True
                }
            self.bins[bin_index].append(item)
            self.map_index_item_in_bins[bin_index].append(global_item_index)
        else:
            # Place item in a new bin
            if item > self.bin_capacity:
                raise ValueError("Item does not fit in a new bin.")
            self.bins.append([item])
            self.map_index_item_in_bins.append([global_item_index])
            reward = -1  # Negative reward for opening a new bin


        # Remove the item from the remaining items
        self.remaining_items.pop(item_index)
        
        if not self.remaining_items:
            self.done = True
        info={
            "bins_used": len(self.bins),
            "order bins": self.bins,
            "item_indices_in_bins": self.map_index_item_in_bins,
            "utilization":self._calculate_utilization()
        } 
        return self._get_state(), reward, self.done, info 

    def _calculate_utilization(self):
        total_capacity = len(self.bins) * self.bin_capacity
        total_used = sum(sum(bin) for bin in self.bins)
        return total_used / total_capacity if total_capacity > 0 else 0.0  
        
    


        

        
                     
        
        


