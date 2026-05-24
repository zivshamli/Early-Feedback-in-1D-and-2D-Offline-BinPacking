from  OneDBinPackingEnv import Offline1DBinPackingEnv

def first_fit_1d_bin_packing(env):
    state = env.reset()
    done = False
    while not done:
        item_index=0
        item=env.remaining_items[item_index]
        placed=False
        for bin_index, bin in enumerate(env.bins):
            if sum(bin) + item <= env.bin_capacity:
                action=(item_index, bin_index)
                state, reward, done, info = env.step(action)
                placed=True
                break
        if not placed:
            action=(item_index, len(env.bins))
            state, reward, done, info = env.step(action)

    return env.bins,info
"""
First-Fit 1D Bin Packing Algorithm:
if __name__ == "__main__":
    items = [4, 8, 1, 4, 2, 1]
    bin_capacity = 10
    env = Offline1DBinPackingEnv(items, bin_capacity)
    bins,info=first_fit_1d_bin_packing(env)
    print("Bins:", bins)
    print("Info:", info)
    """