# RetroAct
This repository is based on our paper: Improving Retrospective Language Agents via Joint Policy Gradient Optimization. It contains the IL dataset we generated, as well as demo code for our fine-tuned planner and reflector.

![image](https://github.com/XueyangFeng/RetroAct/assets/58109619/2ea43cdb-105d-4e5e-aeec-79270db8f0d7)

## Overview
- The Code for different datasets is in `hotpotqa/`, `alfworld/`, and `intercode/`.
  - start IL training by `sft/finetune.sh`
  - start RL training is in `rl/finetune.sh/`
  - start agent test is in `agent/script`
 
## Usage
### Install
You can use following scripts to install related python package through pip:
- git clone https://github.com/XueyangFeng/RetroAct.git
  - pip install -r requirements.txt

### IL training
```
python sft/finetune.py \
    --learning_rate 1e-4 \
    --base_model <your_base_model_path> \
    --data_path <your_sft_data_path> \
    --micro_batch_size 1 \
    --num_epochs 5 \
    --output_path <your_sft_model_path> \
```

### RL training
To reduce the training cost, we use an off-policy approach to train the RL algorithm. You need to first calculate the ref_prob of each token by `rl/ref_prob.py`.

Then, you can start rl training:
```
python rl/finetune.py \
    --learning_rate 1e-4 \
    --base_model <your_sft_model_path> \
    --micro_batch_size 1 \
    --num_epochs 3 \
    --output_path <your_rl_model_path> \
    --regular_coefficient 1.0 \
    --reflector_reward_coefficient 1.0 \
    --clip_advantage False \
    --clip_episode 0.3  \
    --rl_data_path <your_rl_data_path> \
    --regular_data_path <your_regular_data_path>
```

## References
1. Our agent framework code is based on [noahshinn/reflexion](https://github.com/noahshinn/reflexion)
2. Our IL training code is based on [anchen1011/FireAct/](https://github.com/anchen1011/FireAct/)
4. Our RL training code is based on [RUCAIBox/RLMEC](https://github.com/RUCAIBox/RLMEC)
