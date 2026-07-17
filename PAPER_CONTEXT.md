# Terrain Bidding Paper — Context for New Chat Session

## Project
**Title:** "Score Everyone: Incentive-Compatible Task Allocation for Legged Robot Fleets"
**Target:** ICRA 2027 (deadline ~September 2026)
**Repo:** ~/Documents/terrain-bidding (also github.com/ajboyle806/terrain-bidding)

## What the Paper Is About
A fleet of self-interested legged robots receives task assignments from a central planner. Each robot privately predicts execution cost via a learned ensemble model. Robots may underbid (steal tasks) or overbid (avoid work). We design a mechanism making ANY misreport unprofitable.

## The Mechanism (Final Version)
**All-bids + relative + adaptive κ:**
1. Robots submit ensemble predictions for all candidate tasks
2. Planner allocates via Hungarian algorithm with epistemic uncertainty penalty (γ·σ²_epi)
3. After execution, planner scores ALL robots' predictions against realized cost (not just the assigned robot)
4. Scores are relative (subtract fleet mean per task)
5. κ adapts per-task: κ_j = κ_base / max(σ²_epi,j, ε) — automatically higher on familiar terrain, lower on novel terrain
6. Penalty = κ_j · max(0, -relative_score)

**Why each component:**
- All-bids: eliminates selection bias + overbidding loophole
- Relative: creates signal when everyone predicts well
- Adaptive κ: eliminates in-dist exploit without manual tuning

## Key Results (Final Numbers, Adaptive κ)
- In-dist detection: sep=+0.036 (weak but exploit eliminated — offset=0.1 gives gain=-0.15)
- Mild OOD: sep=+2.39, FPR=5.8%
- Moderate OOD: sep=+18.4, FPR=7.2%
- Strong OOD: sep=+62.2, FPR=0.4%
- vs Reputation baseline: 163× better detection, 46× stronger deterrence
- Adversary gain: +200% (vanilla) → -106 (ours)
- Scales: N=4, 8, 16 stable
- Real ANYmal validation: 48 missions, 4074 segments, sep=+0.79, all κ deter
- Collusion: deterrence survives 3:1 adversary majority (gain=-39)
- Symmetric: overbidding penalized equally to underbidding

## Theory
- Proposition 1: Gaussian proper scoring rule → truthful reporting uniquely maximizes expected score
- Proposition 2: κ > R·L·σ²_ale makes honesty dominant strategy
- L=2.26 empirically (shading by 0.3 triples win rate 25%→97%)

## Data Sources
- Simulated: Unitree A1, Isaac Gym, PPO policy, 30K rollouts, 5 terrain types
- Ensemble: 5 CNNs trained on types 0-1, types 2-4 are OOD
- Real: GrandTour dataset (ETH RSL), ANYmal, 48 missions, ZARR from HuggingFace
- All experiment data: outputs/figure_data.pkl
- Figures: figures/*.png (8 figures, regenerated from computed data)

## Code Structure
- terrain_bidding/mechanism/all_bids.py — AllBidsScoringMechanism + AdaptiveKappaMechanism
- terrain_bidding/experiments/compute_figure_data.py — runs all experiments, saves pkl
- terrain_bidding/experiments/plot_figures.py — reads pkl, generates figures
- terrain_bidding/experiments/real_sampler.py — loads ensemble + data, creates PrivateObsTask
- terrain_bidding/grandtour/ — ANYmal data processing pipeline

## Paper Structure (Outline)
I. Introduction (problem, existing fail, our approach, results, contributions)
II. Related Work (task allocation, mechanism design, uncertainty, strategic agents)
III. Problem Formulation (fleet model, information structure, utility, uncertainty decomposition)
IV. Mechanism Design (allocation, why assigned-only fails, all-bids, relative penalty, adaptive κ, propositions)
V. Implementation (policy, collection, ensemble, real data)
VI. Experiments (rational adversary, κ sweep, detection under shift, baselines, scaling, ANYmal)
VII. Discussion (in-dist detection, all-bids assumption, Gaussian, collusion, graceful refusal)
VIII. Conclusion

## What To Do Next
1. Write the paper following the scaffold (detailed bullets exist in chat history)
2. Figures are generated — place them in the paper
3. Key framing: "deterrence-first mechanism with detection as secondary benefit under distribution shift"
4. Don't overclaim detection on in-dist (sep≈0.036 is noisy)
5. The adaptive κ eliminates the exploit — this is a feature to highlight

## Figures Available (figures/*.png)
1. fig1_allbids — detection + FPR across 4 OOD levels (error bars, 3 seeds)
2. fig2_deterrence — κ sweep (in-dist + OOD curves)
3. fig_score_distributions — honest vs adversary score histograms
4. fig_decomposition_terrain — epistemic/aleatoric box plots
5. fig5_reputation — vs reputation baseline
6. fig6_scaling — N=4,8,16
7. fig7_anymal — real ANYmal validation
8. fig_allocation_sensitivity — win rate vs shade δ
