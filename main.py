import argparse
import logging
import sys
from neat import nn, population, statistics, parallel
import math
import numpy as np
import gym
import os
import universe
from scipy import ndimage

from universe import wrappers

logger = logging.getLogger()
ul_x = 170
ul_y = 135
lr_x = 370
lr_y = 335

parser = argparse.ArgumentParser(description='OpenAI Gym Solver')
parser.add_argument('--max-steps', dest='max_steps', type=int, default=10000,
                    help='The max number of steps to take per genome (timeout)')
parser.add_argument('--episodes', type=int, default=100,
                    help="The number of times to run a single genome. This takes the fitness score from the worst run")
parser.add_argument('--render', action='store_true')
parser.add_argument('--generations', type=int, default=50,
                    help="The number of generations to evolve the network")
parser.add_argument('--checkpoint', type=str,
                    help="Uses a checkpoint to start the simulation")
parser.add_argument('--num-cores', dest="numCores", type=int, default=4,
                    help="The number cores on your computer for parallel execution")
args = parser.parse_args()


def blocks(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy//fact * (X//fact) + Y//fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx//fact, sy//fact)
    return res

def simulate_species(net, env, episodes=10, steps=500, render=True):
    fitnesses = []
    for runs in range(episodes):
        inputs = env.reset()
        cum_reward = 0.0
        for j in range(steps):
            if inputs[0] is not None:
                new_obs = np.array(inputs[0]['vision'][ul_y:lr_y, ul_x:lr_x])
                new_obs = new_obs.mean(axis=2)
                new_obs = np.array(blocks(new_obs, 5))
                new_obs = new_obs.flatten()
                outputs = net.serial_activate(new_obs)
            else:
                outputs = env.action_space.sample()
            inputs, reward, done, _ = env.step([env.action_space.sample() for ob in inputs])
            if render:
                env.render()
            if done[0]:
                break
            cum_reward += reward[0]
        fitnesses.append(cum_reward)
    fitness = np.array(fitnesses).mean()
    return fitness

def train_network(env):

    def evaluate_genome(g):
        net = nn.create_feed_forward_phenotype(g)
        return simulate_species(net, env, args.episodes, args.max_steps, render=args.render)

    def eval_fitness(genomes):
        for g in genomes:
            fitness = evaluate_genome(g)
            g.fitness = fitness
    # SIM
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "network_config")
    pop = population.Population(config_path)
    if args.checkpoint:
        pop.load_checkpoint(args.checkpoint)

    # START
    pop.run(eval_fitness, args.generations)

    statistics.save(pop.statistics)
    statistics.save_species_count(pop.statistics)
    statistics.save_species_fitness(pop.statistics)

    winner = pop.statistics.best_genome()

    import pickle
    with open("winner.pkl", 'wb') as output:
        pickle.dump(winner, output, 1)

    raw_input("Enter to start run")
    winner_net = nn.create_feed_forward_phenotype(winner)
    for i in range(100):
        simulate_species(winner_net, env, 1, args.max_steps, render=True)


def main():
    my_env = gym.make('internet.SlitherIOEasy-v0')
    my_env.configure(remotes="vnc://localhost:10000+10001")

    my_env = wrappers.experimental.SafeActionSpace(my_env)
    observation_n = my_env.reset()
    if args.render:
        my_env.render()
    train_network(my_env)


if __name__ == '__main__':
    sys.exit(main())
