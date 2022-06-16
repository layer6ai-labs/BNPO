Bayesian Nonparametrics for Offline Skill Discovery 
===================================================

Code for our ICML 2022 [paper](https://arxiv.org/abs/2202.04675).

# Setup

Create a virtual environment.

```
conda create -n bnp_options python=3.8
conda activate bnp_options
```

Install required packages.

```
pip install numpy gym[atari] matplotlib
conda install pytorch cudatoolkit=10.2 -c pytorch
```

Get the Atari ROMs if you need to run the code on an Atari environment.

```
pip install ale_py
ale-import-roms atari_roms/
```

where atari_roms is the folder containing the ROMs .bin files (they can be downloaded [here](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html)).


# Run proof-of-concept environment

You can run training on the proof-of-concept environment (called room environment) through this command:

```
python train.py --env room --max-epochs 200 --nb-rooms 6 --save-dir poc_env_test
```

which will train in an environment with a vocabulary size of 6 (see environment explanation in the paper) for 200 epochs.
You should obtain a score of 0.996 and a final loss around -3.6.

# Run Atari experiments

Our experiments on the Atari environments consist in pretraining a hierarchical model followed by training a standard RL agent on the augmented enviroment where the skills learned in the pretraining phase can be used as actions.

To train the RL agent, we use the [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) implementation of PPO. We use a custom rollout buffer since we're dealing with an augmented environment. To set it up, clone this [repository](https://github.com/UnrealLink/stable-baselines3) in the parent directory of this one.

```
git clone git@github.com:UnrealLink/stable-baselines3.git ../stable-baselines3
```

You will also need a few additional packages.

```
pip install pandas opencv-python tensorboard
```

You will also need expert trajectories to run the pretraining. The ones for the environments shown in the paper can be found [here](https://drive.google.com/drive/folders/1oDZjhqrxNh0VqeOmz1M9KWWJuM6NZnj0?usp=sharing). They were generated using the [Uber Atari model zoo](https://github.com/uber-research/atari-model-zoo) Apex model.

The following lines of code assume the trajectories were downloaded in the parent directory of this repository.

## Run our nonparametric version

Run the pretraining:

```
python train.py --env atari --demo-file ../trajectories/alien/trajectories.npy --max-epochs 1500 --max-steps 300 --random-seed 0 --batch-size 128 --save-dir runs/atari_pretraining_alien 
```

Run the RL training:

```
python run_atari_exp.py --pretrained-path runs/atari_pretraining_alien --env-name AlienNoFrameskip-v4 --save-dir runs/atari_augmented_alien --training-steps 3000000
```

## Run our fixed-options version

Run the pretraining.

```
python train.py --env atari --demo-file ../trajectories/alien/trajectories.npy --max-epochs 1500 --max-steps 300 --random-seed 0 --batch-size 128 --save-dir runs/atari_pretraining_fixed_alien --fixed-options --K 7
```

Run the RL training.

```
python run_atari_exp.py --pretrained-path runs/atari_pretraining_fixed_alien --env-name AlienNoFrameskip-v4 --save-dir runs/atari_augmented_fixed_alien --training-steps 3000000
```

## Run the compile baseline

The compile baseline code can be found [here](https://github.com/UnrealLink/compile).

```
git clone git@github.com:UnrealLink/compile.git ../compile
```

To run the pretraining:

```
cd ../compile

python train.py --demo-file ../trajectories/alien/trajectories.npy --iterations 1500 --learning-rate 0.001 --latent-dim 7 --num-segments 10 --save-dir runs/atari_pretraining_compile_alien
```

To train the RL agent:

```
cd ../BNPO

python run_atari_exp.py --demo-file ../trajectories/alien/trajectories.npy --pretrained-path runs/atari_pretraining_compile_alien --env-name AlienNoFrameskip-v4 --save-dir ../compile/runs/atari_augmented_compile_alien --training-steps 3000000 --baseline-compile
```

## Run the nonparametric compile baseline

The nonparametric version of compile can be found on the [compile_np] branch of the compile repository.

To run the pretraining:

```
cd ../compile
git checkout compile_np

python train.py --demo-file ../trajectories/alien/trajectories.npy --iterations 1500 --learning-rate 0.001 --latent-dim 1 --num-segments 10 --add-option-interval 10 --save-dir runs/atari_pretraining_compilenp_alien
```

To train the RL agent:

```
cd ../BNPO

python run_atari_exp.py --pretrained-path runs/atari_pretraining_compilenp_alien --env-name AlienNoFrameskip-v4 --save-dir ../compile/runs/atari_augmented_compilenp_alien --training-steps 3000000 --baseline-compile-np
```

## Run the ddo baseline

The ddo baseline code can be found [here](https://github.com/UnrealLink/segment-centroid).

```
git clone git@github.com:UnrealLink/segment-centroid.git ../segment-centroid
```

This codebase is using an older version of python, so you will need to setup another virtual environment.

```
conda create -n ddo_baseline python==3.5
cd ../segment-centroid
pip install -r requirements.txt
conda install pytorch cudatoolkit=10.2 -c pytorch
cd ../BNPO
```

To run the pretraining, use the [ddo_compat]() branch of the repo:

```
git checkout ddo_compat
conda activate ddo_baseline

python train.py --env atari --demo-file ../trajectories/alien/trajectories.npy --max-epochs 10000 --K 7 --baseline-ddo --save-dir runs/atari_pretraining_ddo_alien
```

The stable-baselines3 repository is not compatible with the ddo-baseline repository, so we use a custom pytorch version of ddo to load the trained model to create the augmented environment. 

```
git clone git@github.com:UnrealLink/ddo_baseline_pytorch.git ../ddo_baseline_pytorch
conda activate bnp_options
pip install scipy sklearn
```

You can then launch the RL training.


```
python run_atari_exp.py --pretrained-path runs/atari_pretraining_ddo_alien --env-name AlienNoFrameskip-v4 --save-dir runs/atari_augmented_compile_alien --training-steps 3000000 --baseline-ddo
```