# Standalone SAC example

This directory contains a minimal Soft Actor-Critic implementation extracted
from the rlkit SAC dependency chain and rewritten so it can run without
importing `rlkit`.

Required external dependencies:

- `numpy`
- `torch`
- `pygame` for rendering `TinyContinuousCartPole-v0`

Standard environment dependency:

- `gymnasium` for running environments such as `Pendulum-v1`
- `matplotlib` if you want `--plot-progress`

If `gymnasium` is not installed, the script automatically uses the built-in
`TinyContinuousCartPole-v0` environment.

The default case is `TinyContinuousCartPole-v0`, a continuous-action cart-pole
where the cart moves left/right to balance the pole. This is different from
Gymnasium `CartPole-v1`, which uses discrete left/right actions. The SAC code
here is the continuous-action version, so `TinyContinuousCartPole-v0` is the
matching cart-pole demo.

## Training Cases

### Demo script

Run the complete demo from PowerShell:

```powershell
.\demo_run.ps1 -Epochs 5 -PlotProgress
```

This trains SAC, saves `progress.csv` and checkpoints, optionally draws
`returns.png`, then loads `checkpoints/latest.pt` and runs inference.

Useful variants:

```powershell
.\demo_run.ps1 -Epochs 10 -RenderEval
.\demo_run.ps1 -Epochs 10 -PlotProgress -RenderInfer
.\demo_run.ps1 -ExperimentName test_sac -StepsPerEpoch 500
```

Run the complete demo from Linux/macOS shell:

```bash
chmod +x demo_run.sh
./demo_run.sh --epochs 5 --plot-progress
```

Useful variants:

```bash
./demo_run.sh --epochs 10 --render-eval
./demo_run.sh --epochs 10 --plot-progress --render-infer
./demo_run.sh --experiment-name test_sac --steps-per-epoch 500
```

### Case 1: basic training only

This only trains SAC, evaluates at the end of each epoch, prints metrics to the
terminal, writes `progress.csv`, and saves checkpoints. It does not open a
render window and does not draw figures.

```bash
python sac.py --env TinyContinuousCartPole-v0 --epochs 5
```

Enabled features:

- training
- path collection
- replay buffer updates
- epoch evaluation
- terminal metrics
- `progress.csv`
- model checkpoints

### Case 2: training with evaluation visualization

This trains SAC and opens a render window during evaluation. Use this when you
want to watch how the current policy controls the environment.

```bash
python sac.py --env TinyContinuousCartPole-v0 --epochs 20 --render-eval
```

Enabled features:

- everything in Case 1
- visual evaluation rollout through `render_mode="human"`

### Case 3: training with organized outputs and plotted curves

This trains SAC, saves all experiment outputs under a named folder, and draws
the important return curves into `returns.png`.

```bash
python sac.py --env TinyContinuousCartPole-v0 --epochs 20 --plot-progress --output-dir runs --experiment-name sac_cartpole
```

Enabled features:

- everything in Case 1
- organized output directory
- `returns.png`
- curve plotting from `progress.csv`

Training outputs:

```text
runs/sac_cartpole/
  config.json
  progress.csv
  returns.png                 # created by --plot-progress
  checkpoints/
    latest.pt
    final.pt
    epoch_0001.pt
```

### Case 4: visual training plus plotted outputs

This combines visual evaluation and plotted metrics.

```bash
python sac.py --env TinyContinuousCartPole-v0 --epochs 20 --render-eval --plot-progress --output-dir runs --experiment-name sac_cartpole
```

Enabled features:

- everything in Case 3
- visual evaluation window during training

### Case 5: load a saved policy for inference/control

This does not train. It loads a saved checkpoint and runs deterministic policy
control.

```bash
python infer.py --checkpoint ./runs/sac_cartpole/checkpoints/latest.pt --episodes 5
python infer.py --checkpoint ./runs/sac_cartpole/checkpoints/latest.pt --render
```

Enabled features:

- checkpoint loading
- deterministic policy control
- optional render window with `--render`

If checkpoint loading reports `FileNotFoundError`, use the path printed after
training:

```text
saved final checkpoint to runs/<experiment_name>/checkpoints/latest.pt
```

For older runs made before the cart-pole rename, your checkpoint may be under:

```bash
python infer.py --checkpoint ./runs/sac_pendulum/checkpoints/latest.pt
```

## Terminal Output

Each epoch prints one compact line:

```text
epoch=001 expl_steps=1000 expl_return=-124.82 eval_return=-106.68 expl_reward=-6.241 eval_reward=-5.334 replay_size=2000 q1_loss=... policy_loss=... alpha=...
```

Important fields:

- `expl_return`: mean episode return from exploration paths collected this epoch
- `eval_return`: mean episode return from deterministic evaluation paths
- `expl_reward`: mean single-step reward during exploration
- `eval_reward`: mean single-step reward during evaluation
- `replay_size`: number of transitions stored in replay buffer
- `q1_loss`, `q2_loss`: Q-function regression losses
- `policy_loss`: SAC actor objective value
- `alpha`: entropy temperature

For `TinyContinuousCartPole-v0`, better policies keep the pole upright longer
and move `eval_return` higher.

## Saved Checkpoints

Checkpoints are saved as `.pt` files and can be used by `infer.py` for control.
They contain:

- policy network
- Q networks
- target Q networks
- alpha value
- optimizer states
- config
- observation and action dimensions

## Changing Simulator or Task

The SAC implementation is task-agnostic. To switch to a new simulator or task,
you usually only need to change the environment layer.

### Option 1: use another Gymnasium continuous-control environment

If the environment already exists in Gymnasium and has a continuous action
space, run:

```bash
python sac.py --env YourEnvName-v0 --epochs 20
```

The code automatically reads:

```python
env.observation_space.shape
env.action_space.shape
env.action_space.low
env.action_space.high
```

so the policy and Q networks adapt to the new observation and action
dimensions.

### Option 2: add your own simulator environment

Add a new environment class in `components/envs.py`. The class should provide:

```python
reset() -> observation
step(action) -> next_observation, reward, done, info
close()
observation_space.shape
action_space.shape
action_space.low
action_space.high
action_space.sample()
```

For rendering, also support:

```python
render()
render_mode="human"
```

Then register it in `make_env()`:

```python
def make_env(env_name: str, seed: int, render_mode=None):
    if env_name == "MyRobotTask-v0":
        return MyRobotTaskEnv(seed=seed, render_mode=render_mode)
```

Run it with:

```bash
python sac.py --env MyRobotTask-v0 --epochs 20
```

### What usually changes

- `components/envs.py`: simulator wrapper, observation, action scaling, reward, done, render
- `framework/config.py`: default `env` name if you want to change the default task
- `README.md`: document the new task and recommended command

### What usually does not change

- `components/policies.py`: policy network
- `components/networks.py`: Q networks
- `data/path_collector.py`: path collection
- `data/replay_buffer.py`: replay storage
- `framework/trainer.py`: SAC losses and optimizer updates
- `framework/core_algorithm.py`: training loop

### Important task-design points

- Observation should contain the state needed for control.
- Action space must be continuous for this SAC implementation.
- Reward should increase when behavior improves.
- `done=True` should indicate task failure, success, or episode termination.
- `action_space.low` and `action_space.high` must match the simulator's valid control range.

## File Layout

- `sac.py`: outer training entry point
- `infer.py`: load a checkpoint and run deterministic control
- `components/`: replaceable building blocks
- `components/envs.py`: Gymnasium creation, API compatibility helpers, action scaling
- `components/networks.py`: MLP and Q-function definitions
- `components/policies.py`: tanh-squashed Gaussian policy
- `data/`: path collection and replay storage
- `data/path_collector.py`: rollout complete paths before replay-buffer insertion
- `data/replay_buffer.py`: numpy replay buffer with `add_paths`
- `framework/`: SAC execution framework
- `framework/config.py`: training configuration and argument parsing
- `framework/trainer.py`: SAC losses and optimizer updates
- `framework/core_algorithm.py`: epoch loop, path collection, replay insertion, training, evaluation

The training flow now mirrors the rlkit-style stages:

```text
PathCollector.collect_new_paths()
    -> ReplayBuffer.add_paths(paths)
    -> ReplayBuffer.sample(batch_size)
    -> SACTrainer.train_step(batch)
```
