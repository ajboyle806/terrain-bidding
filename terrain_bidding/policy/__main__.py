"""Entry point for: python -m terrain_bidding.policy"""
import sys

if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
    from terrain_bidding.policy.evaluate import evaluate_policy, check_gate
    path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/policy_best.pt"
    results = evaluate_policy(path)
    passed, issues = check_gate(results)
    for r in results:
        print(f"{'✓' if r.failure_rate < 0.05 else '✗'} {r.terrain:20s} | "
              f"fail={r.failure_rate:.1%} | cost={r.mean_cost:.2f}±{r.std_cost:.2f}")
    print(f"\n{'PASSED' if passed else 'FAILED'}")
    for i in issues:
        print(f"  ⚠ {i}")
else:
    from terrain_bidding.policy.train import train
    train()
