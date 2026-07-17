"""Compute all data needed for paper figures. Saves to outputs/figure_data.pkl.

Run: python3 -m terrain_bidding.experiments.compute_figure_data
"""
import numpy as np
import pickle
from pathlib import Path
from terrain_bidding.experiments.real_sampler import make_real_task_sampler, EnsembleRobot, REAL_ADVERSARY_TYPES
from terrain_bidding.mechanism import FullMechanism, TotalVarianceMechanism, VanillaMechanism, SimConfig, simulate, gaussian_score
from terrain_bidding.mechanism.all_bids import AllBidsScoringMechanism, AdaptiveKappaMechanism
from terrain_bidding.experiments.fleet_experiments import ReputationMechanism
from terrain_bidding.experiments import compute_metrics
from terrain_bidding.configs import MechanismConfig


def run():
    Path("outputs").mkdir(exist_ok=True)

    # Load samplers
    print("Loading samplers...")
    in_dist = make_real_task_sampler(data_path="data/train_types01/test.hdf5")
    ood2 = make_real_task_sampler(data_path="data/ood_type2/test.hdf5")
    ood3 = make_real_task_sampler(data_path="data/ood_type3/test.hdf5")
    ood4 = make_real_task_sampler(data_path="data/ood_type4/test.hdf5")
    anymal = make_real_task_sampler(data_path="data/grandtour/test.hdf5",
                                    ensemble_path="checkpoints/ensemble_grandtour", device="cpu")

    N, M = 4, 2
    terrains = [("in_dist", in_dist), ("ood2", ood2), ("ood3", ood3), ("ood4", ood4)]
    data = {}

    # S_baselines per terrain
    baselines = {}
    for tname, sampler in terrains:
        rng = np.random.default_rng(0)
        scores = [gaussian_score(t[0].true_mu, t[0].robot_observations[0][0].mean(),
                  np.exp(t[0].robot_observations[0][1]).mean())
                  for t in [sampler(1, rng, n_robots=N) for _ in range(200)]]
        baselines[tname] = float(np.mean(scores))
    data["baselines"] = baselines

    # === Fig 1: All-bids vs assigned-only across terrain (3 seeds) ===
    print("Computing fig1 data (3 seeds for error bars)...")
    fig1 = {}
    for tname, sampler in terrains:
        fig1[tname] = {}
        for mn, mech_fn in [("assigned", lambda: FullMechanism()), ("allbids", lambda: AdaptiveKappaMechanism())]:
            seps, fprs = [], []
            for seed in [42, 123, 456]:
                mech_cfg = MechanismConfig(N=N, kappa=20.0, gamma=1.0, S_baseline=0.0)
                fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
                fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))
                def s(n, rng, _sam=sampler): return _sam(M, rng, n_robots=N)
                sim_cfg = SimConfig(mechanism=mech_fn(), mech_cfg=mech_cfg, num_rounds=1000, seed=seed)
                results = simulate(sim_cfg, fleet, s)
                m = compute_metrics(results, 3, None)
                seps.append(m.detection_separation)
                fprs.append(m.false_positive_rate)
            fig1[tname][mn] = {"sep_mean": float(np.mean(seps)), "sep_std": float(np.std(seps)),
                               "fpr_mean": float(np.mean(fprs)), "fpr_std": float(np.std(fprs))}
    data["fig1"] = fig1

    # === Fig 2: κ sweep on IN-DIST (shows crossover) + OOD (all deter) ===
    print("Computing fig2 data (κ sweep, in-dist + OOD)...")
    fig2 = {"in_dist": [], "ood4": []}
    for kappa in [0.5, 1, 2, 5, 10, 20]:
        for tname, sampler, bl_key in [("in_dist", in_dist, "in_dist"), ("ood4", ood4, "ood4")]:
            mech_cfg = MechanismConfig(N=N, kappa=kappa, gamma=1.0, S_baseline=0.0)
            fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
            fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))
            def s(n, rng, _sam=sampler): return _sam(M, rng, n_robots=N)
            sim_cfg = SimConfig(mechanism=AdaptiveKappaMechanism(),
                               mech_cfg=mech_cfg, num_rounds=500, seed=42)
            results = simulate(sim_cfg, fleet, s)
            m = compute_metrics(results, 3, None)
            fig2[tname].append({"kappa": kappa, "gain": m.strategic_gain})
    data["fig2"] = fig2

    # === Fig 3: Score distributions (all-bids on OOD4) ===
    print("Computing fig3 data (score distributions)...")
    fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
    fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))
    rng = np.random.default_rng(42)
    honest_scores, adv_scores = [], []
    for _ in range(2000):
        tasks = ood4(1, rng, n_robots=N)
        t = tasks[0]
        for i, robot in enumerate(fleet):
            bid = robot.bid(t, i, 0)
            mu_hat = bid.mus.mean()
            var_ale = np.exp(bid.log_vars).mean()
            var_epi = ((bid.mus - mu_hat)**2).mean()
            score = gaussian_score(t.true_mu, mu_hat, var_ale + var_epi)
            if i < 3:
                honest_scores.append(score)
            else:
                adv_scores.append(score)
    data["fig3"] = {"honest": honest_scores, "adv": adv_scores}

    # === Fig 4: Epistemic/aleatoric decomposition ===
    print("Computing fig4 data (decomposition)...")
    fig4 = {"epi": [], "ale": []}
    for tname, sampler in terrains:
        epis, ales = [], []
        rng = np.random.default_rng(42)
        for _ in range(200):
            ts = sampler(1, rng, n_robots=N)
            mus, lvs = ts[0].robot_observations[0]
            epis.append(float(((mus - mus.mean())**2).mean()))
            ales.append(float(np.exp(lvs).mean()))
        fig4["epi"].append(epis)
        fig4["ale"].append(ales)
    data["fig4"] = fig4

    # === Fig 5: Reputation comparison (all-bids) ===
    print("Computing fig5 data (reputation)...")
    fig5 = {}
    for mn, mech in [("vanilla", VanillaMechanism()), ("reputation", ReputationMechanism(window=50)),
                     ("allbids", AdaptiveKappaMechanism())]:
        mech_cfg = MechanismConfig(N=N, kappa=20.0, gamma=1.0, S_baseline=0.0)
        fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
        fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))
        def s(n, rng): return ood4(M, rng, n_robots=N)
        sim_cfg = SimConfig(mechanism=mech, mech_cfg=mech_cfg, num_rounds=1000, seed=42)
        results = simulate(sim_cfg, fleet, s)
        m = compute_metrics(results, 3, None)
        fig5[mn] = {"sep": m.detection_separation, "gain": m.strategic_gain, "fpr": m.false_positive_rate}
    data["fig5"] = fig5

    # === Fig 6: Scaling ===
    print("Computing fig6 data (scaling)...")
    fig6 = []
    for fleet_N in [4, 8, 16]:
        fleet_M = fleet_N // 2
        n_adv = fleet_N // 4
        n_honest = fleet_N - n_adv
        mech_cfg = MechanismConfig(N=fleet_N, kappa=20.0, gamma=1.0, S_baseline=0.0)
        for mn, mech in [("vanilla", VanillaMechanism()), ("allbids", AdaptiveKappaMechanism())]:
            fleet = [EnsembleRobot(robot_id=i) for i in range(n_honest)]
            for i in range(n_adv):
                fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=n_honest+i, offset=0.5))
            def s(n, rng): return ood4(fleet_M, rng, n_robots=fleet_N)
            sim_cfg = SimConfig(mechanism=mech, mech_cfg=mech_cfg, num_rounds=500, seed=42)
            results = simulate(sim_cfg, fleet, s)
            m = compute_metrics(results, n_honest, None)
            fig6.append({"N": fleet_N, "mech": mn, "gain": m.strategic_gain, "sep": m.detection_separation})
    data["fig6"] = fig6

    # === Fig 7: ANYmal ===
    print("Computing fig7 data (ANYmal)...")
    rng = np.random.default_rng(0)
    scores_bl = [gaussian_score(t[0].true_mu, t[0].robot_observations[0][0].mean(),
                 np.exp(t[0].robot_observations[0][1]).mean())
                 for t in [anymal(1, rng, n_robots=N) for _ in range(200)]]
    S_bl_anymal = float(np.mean(scores_bl))
    fig7 = {"S_baseline": S_bl_anymal, "mechanisms": {}, "kappa_sweep": []}
    for mn, mech in [("vanilla", VanillaMechanism()), ("allbids", AdaptiveKappaMechanism())]:
        mech_cfg = MechanismConfig(N=N, kappa=20.0, gamma=1.0, S_baseline=0.0)
        fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
        fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))
        def s(n, rng): return anymal(M, rng, n_robots=N)
        sim_cfg = SimConfig(mechanism=mech, mech_cfg=mech_cfg, num_rounds=1000, seed=42)
        results = simulate(sim_cfg, fleet, s)
        m = compute_metrics(results, 3, None)
        fig7["mechanisms"][mn] = {"sep": m.detection_separation, "fpr": m.false_positive_rate, "gain": m.strategic_gain}
    for kappa in [0.5, 1, 2, 5, 10, 20]:
        mech_cfg = MechanismConfig(N=N, kappa=kappa, gamma=1.0, S_baseline=0.0)
        fleet = [EnsembleRobot(robot_id=i) for i in range(3)]
        fleet.append(REAL_ADVERSARY_TYPES["fixed_offset"](robot_id=3, offset=0.5))
        def s(n, rng): return anymal(M, rng, n_robots=N)
        sim_cfg = SimConfig(mechanism=AdaptiveKappaMechanism(),
                           mech_cfg=mech_cfg, num_rounds=500, seed=42)
        results = simulate(sim_cfg, fleet, s)
        m = compute_metrics(results, 3, None)
        fig7["kappa_sweep"].append({"kappa": kappa, "gain": m.strategic_gain})
    data["fig7"] = fig7

    # === Fig 8: Allocation sensitivity ===
    print("Computing fig8 data (allocation sensitivity)...")
    from terrain_bidding.mechanism import Bid
    offsets = np.linspace(0, 1.5, 15).tolist()
    win_rates = []
    rng = np.random.default_rng(42)
    mech_cfg = MechanismConfig(N=N, kappa=20.0, gamma=1.0, S_baseline=0.0)
    for offset in offsets:
        wins = 0
        for _ in range(500):
            tasks = ood4(1, rng, n_robots=N)
            bids = []
            for i in range(N):
                mus, lvs = tasks[0].robot_observations[i]
                if i == 3:
                    mus = mus - offset
                bids.append([Bid(robot_id=i, task_id=0, mus=mus, log_vars=lvs)])
            a = FullMechanism().allocate(bids, mech_cfg)
            if a[3] >= 0:
                wins += 1
        win_rates.append(wins / 500)
    data["fig8"] = {"offsets": offsets, "win_rates": win_rates}

    # Save
    with open("outputs/figure_data.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"\nAll data saved to outputs/figure_data.pkl")


if __name__ == "__main__":
    run()
