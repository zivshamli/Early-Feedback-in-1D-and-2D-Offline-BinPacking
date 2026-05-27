from BestFit1D import best_fit_1d_bin_packing
from FirstFit1D import first_fit_1d_bin_packing
from OneDBinPackingEnv import Offline1DBinPackingEnv
from ORLibLoader import ORLib1DBinPackingLoader
from BinPacking1DMetric import BinPackingMetrics

ORLib_loader = ORLib1DBinPackingLoader("binpack1.txt")
instances = ORLib_loader.load()
results_first_fit = []
results_best_fit = []

for instance in instances: 
    print(f"Processing instance: {instance.name}")
    env = Offline1DBinPackingEnv(instance.items, instance.capacity)
    metrics = BinPackingMetrics(instance.capacity, instance.optimal_bins) 


    bins_first_fit, info_first_fit = first_fit_1d_bin_packing(env)
    print(f"name: {instance.name}, First-Fit Bins: {bins_first_fit}, Info: {info_first_fit}")
    results_first_fit.append(metrics.evaluate_episode(bins_first_fit,instance.num_items))
    
    
    env.reset()  # Reset environment for the next algorithm
    bins_best_fit, info_best_fit = best_fit_1d_bin_packing(env)
    print(f"name: {instance.name}, Best-Fit Bins: {bins_best_fit}, Info: {info_best_fit}")
    results_best_fit.append(metrics.evaluate_episode(bins_best_fit,instance.num_items))
    print("-" * 50)

print("Summary of Results:")
avg_bins_first_fit = sum(result["optimality_gap"] for result in results_first_fit) / len(results_first_fit)
avg_bins_best_fit = sum(result["optimality_gap"] for result in results_best_fit) / len(results_best_fit)

print(f"Optimality Gap - First-Fit: {avg_bins_first_fit}")
print(f"Optimality Gap - Best-Fit: {avg_bins_best_fit}")

print("-"*50)


avg_utilization_first_fit = sum(result["utilization"] for result in results_first_fit) / len(results_first_fit)
avg_utilization_best_fit = sum(result["utilization"] for result in results_best_fit) / len(results_best_fit)

print(f"Average Utilization - First-Fit: {avg_utilization_first_fit}")
print(f"Average Utilization - Best-Fit: {avg_utilization_best_fit}")
