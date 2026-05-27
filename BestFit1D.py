from OneDBinPackingEnv import Offline1DBinPackingEnv

def best_fit_1d_bin_packing(env):
    state = env.reset()
    done = False
    global_item_index=0
    while not done:
        item_index=0
        item=env.remaining_items[item_index]
        best_bin_index=None
        min_space_left=env.bin_capacity+1
        for bin_index, bin in enumerate(env.bins):
            space_left=env.bin_capacity-sum(bin)
            if space_left>=item and space_left<min_space_left:
                best_bin_index=bin_index
                min_space_left=space_left
        if best_bin_index is not None:
            action=(item_index, best_bin_index)
            state, reward, done, info = env.step(action,global_item_index)
        else:
            action=(item_index, len(env.bins))
            state, reward, done, info = env.step(action,global_item_index)
        global_item_index+=1    

    return env.bins,info

if __name__ == "__main__":
    items = [4, 8, 1, 4, 2, 1]
    bin_capacity = 10
    env = Offline1DBinPackingEnv(items, bin_capacity)
    bins,info=best_fit_1d_bin_packing(env)
    print("Bins:", bins)
    print("Info:", info)
   