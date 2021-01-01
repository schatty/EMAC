from datetime import datetime
import argparse
import shutil
import os

from models.trainer import Trainer


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
        parser.add_argument("--env", default="Walker2d-v3")          # OpenAI gym environment name
        parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
        parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
        parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
        parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
        parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
        parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
        parser.add_argument("--discount", default=0.99)                 # Discount factor
        parser.add_argument("--tau", type=float, default=0.005)                     # Target network update rate
        parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
        parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
        parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
        parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
        parser.add_argument("--save_buffer", default=0)
        parser.add_argument("--save_memory", default=False, action="store_true")
        parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--save_model_every", type=int, default=1000000)      # Save model every timesteps
        parser.add_argument("--exp_name", default="test")
        parser.add_argument("--results_dir", default="/home/igor/results_dec_29_noise") # Directory for storing all experimental data
        parser.add_argument("--ep_len", default=1000, type=int) # Length of the episdoe
        parser.add_argument("--alpha", default=0.1, type=float)
        parser.add_argument("--mem_capacity", default=100000, type=int) # Number of transitions that memory holds
        parser.add_argument("--k", default=2, type=int) # K-nearest for the memory retrieval
        parser.add_argument("--weak_memory", default=1, type=int)
        parser.add_argument("--substeps", default=1, type=int) # Number of train ops per transition from env
        args = parser.parse_args()

        print("---------------------------------------")
        print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")

        dt = datetime.now()
        results_dir = args.results_dir
        exp_dir = dt.strftime("%b_%d_%Y")
        exp_dir = f"{results_dir}/{exp_dir}_{args.policy}_{args.env}_{args.seed}_{args.exp_name}"
        if os.path.exists(exp_dir):
            ans = input(f"Directory {exp_dir} already exists. Overwrite? [Y/n] ")
            if ans == "Y":
                shutil.rmtree(exp_dir)
            else:
                raise Exception("Trying to rewrite existing experiment. Exiting...")
        print(f"Saving dir: {exp_dir}")

        folders = ["models", "buffers", "tb"]
        for fold in folders:
            fn = f"{exp_dir}/{fold}"
            if not os.path.exists(fn):
                os.makedirs(fn)

        config = vars(args)
        print("Config: ", config)

        trainer = Trainer(config)
        trainer.train(exp_dir)

