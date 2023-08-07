# meta-qlearning-humanoid
Meta QLearning Experiments on Humanoid Robot

## Conducted experiments:
### Learn Stepping using MQL
Test how adaptable the humanoid is by performing:
- Side stepping
- Ascending and Descending

## Simulation Platform
- JVRC Humanoid robot
- Simulation environment: Wrote packages to import JVRC to Mujoco and Gym environments

## Setting up the environment:
This repository contains everything needed to set up the environment and get the simulation up and running. 

### Clone the repository: 
`git clone <repo name>`
Make sure the file structure is as follows:
```
<Your folder>
├── algs
│   └── MQL
│       ├── buffer.py
│       └── mql.py
├── configs
│   └── abl_envs.json
├── Humanoid_environment
│   ├── envs
│   │   ├── common
│   │   └── jvrc
│   ├── models
│   │   ├── cassie_mj_description
│   │   └── jvrc_mj_description
│   ├── scripts
│   │   ├── debug_stepper.py
│   │   └── plot_logs.py
│   ├── tasks
│   │   │   ├── rewards.cpython-37.pyc
│   │   │   ├── stepping_task.cpython-37.pyc
│   │   │   └── walking_task.cpython-37.pyc
│   │   ├── rewards.py
│   │   ├── stepping_task.py
│   │   └── walking_task.py
│   └── utils
│       └── footstep_plans.txt
├── misc
│   ├── env_meta.py
│   ├── logger.py
│   ├── runner_meta_offpolicy.py
│   ├── runner_multi_snapshot.py
│   ├── torch_utility.py
│   └── utils.py
├── models
│   ├── networks.py
│   └── run.py
├── README.md
└── run_script.py
```

### Installing packages:
```
pip3 install -r requirements.txt
```

### Training
```
python3 run_script.py
```
Please note: Code is written to train using a GPU

Training time: ~55 hours on RTX 3080

##### Feel free to contact the author for pre-trained model

## References:
Rasool Fakoor, Pratik Chaudhari, Stefano Soatto, & Alex Smola (2020). Meta-Q-Learning. In ICLR 2020, Microsoft Research Reinforcement Learning Day 2021

