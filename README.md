# RetroAct
This repository is based on our paper: Improving Retrospective Language Agents via Joint Policy Gradient Optimization. It contains the IL dataset we generated, as well as demo code for our fine-tuned planner and reflector.

![image](https://github.com/XueyangFeng/RetroAct/assets/58109619/2ea43cdb-105d-4e5e-aeec-79270db8f0d7)

## Overview
- The Code for different datasets is in `hotpotqa/`, `alfworld/`, and `intercode/`.
  - start IL training by `sft/finetune.sh`
  - start RL training is in `rl/finetune.sh/`
  - start agent test is in `agent/script`
 
## Usage
You can use following scripts to install related python package through pip:
- git clone https://github.com/XueyangFeng/RetroAct.git
  - cd ReHAC
  - pip install -r requirements.txt
