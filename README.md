# Crazy Car with Reinforcement Learning

## Setup Environment

You need to install [pipenv](https://github.com/pypa/pipenv) for create virtual environment and install depedencies package.

```zsh
pipenv shell
```

## Running the code

training a model.

```zsh
python -m pysim.scripts.train --load=PATH --nupdate=100 OUTPUTMODEL
```

`load` is a optional argument if you set `PATH` to somepath. You will load model from `PATH`.

`nupdate` default is 100. This argument indicate number of update.

`OUTPUTMODEL` is the name of output.

testing a model.

```zsh
python -m pysim.scripts.test PATH2MODEL
```

`PATH2MODEL` is path to model that you want to test.

manual control.

```zsh
python -m pysim.scripts.control
```
