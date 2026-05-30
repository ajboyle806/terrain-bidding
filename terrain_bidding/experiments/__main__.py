import pickle
from terrain_bidding.experiments import run_experiment_grid

results = run_experiment_grid()

# Save results
with open("outputs/experiment_results.pkl", "wb") as f:
    pickle.dump(results, f)

# Print summary
print(f"\n{'='*80}")
print(f"{'Condition':<55} {'Eff':>6} {'Gain':>7} {'Sep':>6} {'FPR':>5}")
print('-'*80)
for c, m in sorted(results.items()):
    print(f"{c:<55} {m.allocation_efficiency:>6.3f} {m.strategic_gain:>7.1%} "
          f"{m.detection_separation:>6.3f} {m.false_positive_rate:>5.3f}")
