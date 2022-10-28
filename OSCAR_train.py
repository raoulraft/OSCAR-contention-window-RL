import numpy as np
import torch
import gym
import argparse
import os
import wandb
import tqdm
import subprocess
import matplotlib.pyplot as plt
import time


from ns3gym import ns3env
from ns3gym.start_sim import find_waf_path

from agents.our_ddpg.utils import ReplayBuffer
from agents.our_ddpg.preprocessor import Preprocessor

from agents.our_ddpg.Our_DDPG import DDPG
from agents.our_ddpg.loggers import Logger

from exceptions import AlreadyRunningException
from wrappers import EnvWrapper


if __name__ == "__main__":

	# check scenario.h to see and edit the scenarios
	scenario = "basic"  # convergence
	agent_being_trained = "OurDDPG"

	simTime = 15
	stepTime = 0.01
	history_length = 300
	steps_per_ep = int(simTime/stepTime)
	EPISODE_COUNT = 12

	for nWifi in [40, 45, 50]:

		sim_args = {
		    "simTime": simTime,
		    "envStepTime": stepTime,
		    "historyLength": history_length,
		    "agentType": "continuous",
		    "scenario": scenario,
		    "nWifi": nWifi,
		}
		tags = ["Rew: normalized speed",
		        "OurDDPG",
		        sim_args['scenario'],
		        f"Actor: UNDEFINED",
		        f"Critic: UNDEFINED",
		        f"Instances: 1",
		        f"Station count: {sim_args['nWifi']}",
		        *[f"{key}: {sim_args[key]}" for key in list(sim_args)[:3]]]

		wtags = [f"{EPISODE_COUNT}ep training", f"{simTime}s", f"{nWifi} nWifi", f"envStep {stepTime}", agent_being_trained, "train"]

		run = wandb.init(name=f"{nWifi} OurDDPG No Mean Window|No Std Window 1 OBS [128x128]", entity="xraulz", project="contention_window", tags = wtags, reinit=True)

		logger = Logger(False, tags, None, experiment=None)
		logger.begin_logging(EPISODE_COUNT, steps_per_ep, None, None, stepTime)
		preprocess = Preprocessor(False).preprocess

		print("Steps per episode:", steps_per_ep)

		threads_no = 1
		env = EnvWrapper(threads_no, **sim_args)
		env.reset()


		parser = argparse.ArgumentParser()
		parser.add_argument("--policy", default="OurDDPG")                  # Policy name (TD3, DDPG or OurDDPG)
		parser.add_argument("--env", default="ContentionWindow")          # OpenAI gym environment name
		parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
		parser.add_argument("--start_timesteps", default=300, type=int)# Time steps initial random policy is used
		parser.add_argument("--eval_freq", default=10e10, type=int)       # How often (time steps) we evaluate
		parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
		parser.add_argument("--expl_noise", default=0.1) #0.1         # Std of Gaussian exploration noise
		parser.add_argument("--batch_size", default=32, type=int) # 256      # Batch size for both actor and critic
		parser.add_argument("--discount", default=0.7) #0.99             # Discount factor
		parser.add_argument("--tau", default=1e-3) # 0.005                    # Target network update rate
		parser.add_argument("--policy_noise", default=0.2) # R:TD3 only              # Noise added to target policy during critic update
		parser.add_argument("--noise_clip", default=0.5) # R:TD3 only                # Range to clip target policy noise
		parser.add_argument("--policy_freq", default=2, type=int) # R:TD3 only       # Frequency of delayed policy updates
		parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
		parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
		args = parser.parse_args()

		file_name = f"{args.policy}_{args.env}_{args.seed}"
		print("---------------------------------------")
		print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
		print("---------------------------------------")

		if not os.path.exists("./results"):
			os.makedirs("./results")

		if args.save_model and not os.path.exists("./models"):
			os.makedirs("./models")

		torch.manual_seed(args.seed)
		np.random.seed(args.seed)

		state_dim = 1
		action_dim = 1
		max_action = 1  # DDPG outputs action from -1 to 1
		real_max_action = 6  # actions is scaled to be in range from 0 to 6
		stateSize = state_dim


		kwargs = {
			"state_dim": state_dim,
			"action_dim": action_dim,
			"max_action": max_action,
			"discount": args.discount,
			"tau": args.tau,
		}

		# Initialize policy
		if args.policy == "OurDDPG":
			policy = DDPG(**kwargs)
		else:
			print("Policy not available")

		if args.load_model != "":
			policy_file = file_name if args.load_model == "default" else args.load_model
			policy.load(f"./models/{policy_file}")

		replay_buffer = ReplayBuffer(state_dim, action_dim)


		state, done = env.reset(), False
		episode_reward = 0
		episode_timesteps = 0
		time_step = 0
		for episode in range(EPISODE_COUNT):
			episode_reward = 0

			try:
				env.run()
			except AlreadyRunningException as e:
				pass

			sent_mb = 0
			obs_dim = 1
			state = env.reset()
			state = state[0][:stateSize]
			state = np.reshape(state, stateSize)
			last_action = None
			with tqdm.trange(1, steps_per_ep+1) as t:
				for step in t: #

					# Select action randomly or according to policy
					if time_step < args.start_timesteps:
						action = np.array([[np.random.uniform(0,6)]])
						real_action = action
					else:
						action = (
							policy.select_action(np.array(state))
							+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
						).clip(-max_action, max_action)
						real_action = real_max_action * (action + 1) / 2   # scale it from -1,1 to 0,6

						real_action = np.array([real_action])

					# Perform action
					next_state, reward, done, info = env.step(real_action)  # inserting action between 0 and 6
					not_processed_state = preprocess(np.reshape(next_state, (-1, len(env.envs), obs_dim)))

					next_state = next_state[0][:stateSize]
					next_state = np.reshape(next_state, stateSize)


					# Store data in replay buffer
					replay_buffer.add(state, action, next_state, reward, done)  # inserting action between -1 and 1

					state = next_state
					episode_reward += reward
					if step > 300:
						loss = {"actor_loss": policy.actor_loss, "critic_loss": policy.critic_loss}
						logger.log_round(state, reward, episode_reward, info, loss, np.mean(not_processed_state, axis=0)[0], episode*steps_per_ep+step)

					t.set_postfix(mb_sent=f"{logger.sent_mb:.2f} Mb", curr_speed=f"{logger.current_speed:.2f} Mbps")


					# Train agent after collecting sufficient data
					if time_step >= args.start_timesteps:
						policy.train(replay_buffer, args.batch_size)

					time_step += 1

					if done:
						print(f"Total T: {t+1} Episode Num: {episode} Episode T: {t} Reward: {episode_reward:.3f}")

						break
			logger.log_episode(episode_reward, logger.sent_mb/(simTime), episode)
			env.close()
		run.finish()
