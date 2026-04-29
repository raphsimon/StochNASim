# Learning Robust Penetration Testing Policies under Partial Observability

Code accompanying the paper **["Learning Robust Penetration Testing Policies under Partial Observability: A Systematic Evaluation"](https://openreview.net/forum?id=YkUV7wfk19)** (Simon, Libin, Mees), published in _Transactions on Machine Learning Research_ (TMLR), 2026.

This repository contains:

- **StochNASim**, a stochastic, partially observable extension of [NASim](https://github.com/Jjschwartz/NetworkAttackSimulator) that regenerates the network topology, host properties, and action space at every episode, and supports networks of variable size.
- Implementations of PPO-TrXL, adapted to StochNASim.

## Overview

We model penetration testing as a partially observable, stochastic sequential decision-making problem over networks of varying size. We compare a PPO baseline against approaches designed to mitigate partial observability — frame-stacking (PPO-FS), observation augmentation (PPO-AO), recurrent networks (PPO-LSTM), and Transformer-XL (PPO-TrXL). Our findings show that simple history aggregation via observation augmentation outperforms more complex memory architectures, converging up to four times faster while learning more interpretable policies.

## What StochNASim adds over NASim

|Feature|NASim|StochNASim|
|---|---|---|
|Network topology|Fixed per scenario|Regenerated each episode|
|Network size|Fixed (e.g., 5 or 8 hosts)|Variable (e.g., 5–8 hosts)|
|Initial state|Single fixed state|Distribution of initial states|
|Host properties (OS, services, processes)|Static|Regenerated each reset|
|Observation space|Fixed `(m_c + 1) × n`|Variable `(m + 1) × n`|
|Action space|Fixed per scenario|Regenerated each reset (padded with No-Op)|
|Stochasticity|Action success only|Action success + network generation|

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/raphsimon/StochNASim.git
cd StochNASim
pip install -e .
```

## Quick start

```python
import gymnasium as gym
import nasim   # registers StochNASim environments

env = gym.make(
    'StochPO-v0',
    min_num_hosts=5,
    max_num_hosts=8,
    exploit_probs=0.9,
    privesc_probs=0.9,
    seed=2,
    render_mode='human',
)

obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

## Repository structure

```
nasim/                   Original NASim source, preserved from upstream
nasim/stochastic_envs/   StochNASim environment (extends NASim) + training code
nasim/agents/ppo_trxl/   Adapted implementation of PPO-TrXL for StochNASim 
test/                    Test scripts for the environment
configs/                 Best hyperparameters per algorithm (paper Appendix A)
docs/                    pre-existing documentation
```

For the underlying NASim documentation, see [https://networkattacksimulator.readthedocs.io/](https://networkattacksimulator.readthedocs.io/).

## PPO-TrXL

We also provide an adapted implementation of PPO-TrXL to StochNASim. With hyperparameter tuning.

The basis we use comes from [CleanRL](https://docs.cleanrl.dev/rl-algorithms/ppo-trxl/). PPO-TrXL was created and implemented by Marco Plaines et al. for their work titled: [Memory Gym: Towards Endless Tasks to Benchmark Memory Capabilities of Agents](https://arxiv.org/abs/2309.17207).

### Exmaple Usage
```
cd nasim/agents/ppo_trxl && python ppo_trxl.py \
        --exp-name smoke_test \
        --env-id StochPO-v0 \
        --num-envs 2 \
        --num-steps 64 \
        --total-timesteps 10000 \
        --num-evals 1 \
        --eval-freq 5000 \
        --num-eval-envs 2 \
        --num-eval-episodes 4 \
        --anneal-steps 256 \
        --trxl-memory-length 32 \
        --trxl-num-layers 2 \
        --trxl-dim 64 \
        --trxl-num-heads 1 \
        --no-cuda
```

### Hyperparameter Tuning
```
cd nasim/agents/ppo_trxl && python hyperparams_search.py \
        --env-id StochPO-v0 \
        --num-envs 8 \
        --num-steps 768 \
        --total-timesteps 5000000 \
        --db-url <place URL to Optune database here>\
        --trials  75 \
        --max-total-trials 250 \
        --study-name ppo_trxl_genpo \
        --pruner-warmup-steps 1900000 \
        --num-evals 5 \
        --num-eval-envs 8 \
        --num-eval-episodes 100 \
        --anneal-steps 4020000
```

## Citation

If you use StochNASim or this code, please cite:

```bibtex
@article{simon2026learning,
  title   = {Learning Robust Penetration Testing Policies under Partial Observability: A Systematic Evaluation},
  author  = {Simon, Raphael and Libin, Pieter and Mees, Wim},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=YkUV7wfk19}
}
```

We also recommend citing the underlying NASim:

```bibtex
@misc{schwartz2019nasim,
  title        = {NASim: Network Attack Simulator},
  author       = {Schwartz, Jonathon and Kurniawatti, Hanna},
  year         = {2019},
  howpublished = {\url{https://networkattacksimulator.readthedocs.io/}}
}
```

## Acknowledgments

StochNASim is built on top of [NASim](https://github.com/Jjschwartz/NetworkAttackSimulator) by Jonathon Schwartz and Hanna Kurniawatti, released under the MIT License. We thank the NASim authors for providing the foundation that made this work possible.

PPO-TrXL is adapted from [cleanRL](https://github.com/vwxyzjn/cleanrl) (Huang et al., 2022). PPO, PPO-FS, PPO-AO, and PPO-LSTM use [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) v2.4 and [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) for hyperparameter tuning. The Transformer-XL implementation follows Pleines et al. (2025).

This research was funded by the Royal Higher Institute of Defence under the project DAP23/05. This work was supported by Flemish Government under the "Onderzoeksprogramma Artificiële Intelligentie (AI) Vlaanderen" program. The resources and services used in this work were, in part, provided by the VSC (Flemish Supercomputer Center), funded by the Research Foundation – Flanders (FWO) and the Flemish Government. Pieter Libin acknowledges support from the Research council of the Vrije Universiteit Brussel (OZR-VUB) via grant number OZR3863BOF.

## License

This project is released under the MIT License. The original NASim codebase, which this work extends, is also MIT-licensed and copyright © 2020 Jonathon Schwartz; both copyright notices are preserved in the `LICENSE` file.
