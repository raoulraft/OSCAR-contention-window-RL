from ns3gym import ns3env
import tqdm
import subprocess
from collections import deque
import numpy as np
import wandb

from agents.ddpg.agent import Agent, Config
from agents.ddpg.model import Actor
from agents.ddpg.teacher import Teacher, EnvWrapper
from agents.ddpg.preprocessor import Preprocessor

"""
Training of the CCOD algorithm [see: https://github.com/wwydmanski/RLinWiFi]
"""

# check scenario.h to see and edit the scenarios
scenario = "basic"  # convergence
agent_being_trained = "CCOD"

simTime = 15
stepTime = 0.01
nWifi = 25
history_length = 300

EPISODE_COUNT = 12
steps_per_ep = int(simTime/stepTime)

sim_args = {
    "simTime": simTime,
    "envStepTime": stepTime,
    "historyLength": history_length,
    "agentType": Agent.TYPE,
    "scenario": scenario,
    "nWifi": nWifi,
}

wtags = [f"{EPISODE_COUNT}ep training", f"{simTime}s", f"{nWifi} nWifi", f"envStep {stepTime}", agent_being_trained, "train"]

print("Steps per episode:", steps_per_ep)

threads_no = 1
env = EnvWrapper(threads_no, **sim_args)

env.reset()
ob_space = env.observation_space
ac_space = env.action_space

print("Observation space shape:", ob_space)
print("Action space shape:", ac_space)

assert ob_space is not None

teacher = Teacher(env, 1, Preprocessor(False))

lr_actor = 4e-4
lr_critic = 4e-3

config = Config(buffer_size=4*steps_per_ep*threads_no, batch_size=32, gamma=0.7, tau=1e-3, lr_actor=lr_actor, lr_critic=lr_critic, update_every=1)
agent = Agent(history_length, action_size=1, config=config, actor_layers=[8, 128, 16], critic_layers=[8,128,16])

# Test the model
hyperparams = {**config.__dict__, **sim_args}
tags = ["Rew: normalized speed",
        f"{Agent.NAME}",
        sim_args['scenario'],
        f"Actor: {lr_actor}",
        f"Critic: {lr_critic}",
        f"Instances: {threads_no}",
        f"Station count: {sim_args['nWifi']}",
        *[f"{key}: {sim_args[key]}" for key in list(sim_args)[:3]]]


run = wandb.init(entity="xraulz", project="contention_window", tags = wtags)

logger = teacher.train(agent, EPISODE_COUNT,
                        simTime=simTime,
                        stepTime=stepTime,
                        history_length=history_length,
                        send_logs=True,
                        experimental=True,
                        tags=tags,
                        parameters=hyperparams)
