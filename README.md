# Soving Continuous Control with Episodic Memory

PyTorch implementation of Episodic Memory Actor-Critic (EMAC).

![alt text](https://user-images.githubusercontent.com/23639048/116824091-fd67a300-ab90-11eb-9eab-d589fe8ed3bd.png)

TD3 and DDPG architecture parameters were based on official TD3 implementation: [link](https://github.com/sfujim/TD3)

## Usage

For training run:

`python train.py --policy EMAC --env Walker2d-v3 --k 2 --alpha 0.1 --max_timesteps 200000 --device cuda:0`

## Results

Paper training curves can be found in `curves` directory as saved TensorBoard logs in json format. 
