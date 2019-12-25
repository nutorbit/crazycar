# Crazy Car with Reinforcement Learning

## Setup Environment

```zsh
pip install -r requirements.txt
```

## Running the code

**training a model.**

```zsh
python -m pysim.scripts.train --iters=NUM idx --description="test" name
```

`iters` is a number of episode (`NUM`) for training the agent.

`idx` is a number of experiments.

`description` is a word to describe the experiment.

`name` is a model name (`ppo1`, `sac`, `td3`, `ddpg`).


**testing a model.**

```zsh
python -m pysim.scripts.test model PATH
```

`model` is a model name (`ppo1`, `sac`, `td3`, `ddpg).

`PATH` is path to model that you want to test.