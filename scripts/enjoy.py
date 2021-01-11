#!/usr/bin/env python3

"""
Visualize the performance of a model on a given environment.
"""
import os
import sys
import argparse
import gym
import time
sys.path.insert(0, os.environ['BABYAI_ROOT'])
sys.path.insert(0, os.path.join(os.environ['BABYAI_ROOT'], 'babyai'))
import babyai.utils as utils
from PIL import Image, ImageDraw

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin or --demos REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or --model demos-origin required)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model or --demos REQUIRED)")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (default: 0 if model agent, 1 if demo agent)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--pause", type=float, default=0.1,
                    help="the pause between two consequent actions of an agent")
parser.add_argument("--manual-mode", action="store_true", default=False,
                    help="Allows you to take control of the agent at any point of time")
parser.add_argument("--save-loc", default="vis",
                    help="What folder to save the visualization in")
parser.add_argument("--samples", type=int, default=1,
                    help="How many episodes to run for")

args = parser.parse_args()

action_map = {
    "LEFT"   : "left",
    "RIGHT"  : "right",
    "UP"     : "forward",
    "PAGE_UP": "pickup",
    "PAGE_DOWN": "drop",
    "SPACE": "toggle"
}

assert args.model is not None or args.demos is not None, "--model or --demos must be specified."
if args.seed is None:
    args.seed = 0 if args.model is not None else 1

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)

global obs
obs = env.reset()
print("Mission: {}".format(obs["mission"]))

# Define agent
agent = utils.load_agent(env, args.model, args.demos, args.demos_origin, args.argmax, args.env)

# Run the agent

done = True

action = None

try:
    os.makedirs(args.save_loc)
except OSError as e:
    pass

def keyDownCb(keyName):
    global obs
    # Avoiding processing of observation by agent for wrong key clicks
    if keyName not in action_map and keyName != "RETURN":
        return

    agent_action = agent.act(obs)['action']

    if keyName in action_map:
        action = env.actions[action_map[keyName]]

    elif keyName == "RETURN":
        action = agent_action

    obs, reward, done, _ = env.step(action)
    agent.analyze_feedback(reward, done)
    if done:
        print("Reward:", reward)
        obs = env.reset()
        print("Mission: {}".format(obs["mission"]))

def writeMission(sample, mission):
    txt = Image.new('RGB', sample.size, (0,0,0))
    d = ImageDraw.Draw(txt)
    d.text((20,20), mission, fill=(255,255,255))
    return txt

step = 1
episode_num = 0
while episode_num < args.samples:
    save_location = args.save_loc + "/" + str(episode_num)
    try:
        os.makedirs(save_location)
    except OSError as e:
        pass
    time.sleep(args.pause)
    renderer = env.render("rgb_array")
    img = Image.fromarray(renderer, 'RGB')
    img.save(save_location + '/' + str(step) + '.png')

    if args.manual_mode and renderer.window is not None:
        renderer.window.setKeyDownCb(keyDownCb)
    else:
        result = agent.act(obs)
        obs, reward, done, _ = env.step(result['action'])
        agent.analyze_feedback(reward, done)
        if 'dist' in result and 'value' in result:
            dist, value = result['dist'], result['value']
            dist_str = ", ".join("{:.4f}".format(float(p)) for p in dist.probs[0])
            print("step: {}, mission: {}, action: {}, dist: {}, entropy: {:.2f}, value: {:.2f}".format(
                step, obs["mission"], result['action'], dist_str, float(dist.entropy()), float(value)))
        else:
            print("step: {}, mission: {}".format(step, obs['mission']))
        if done:
            print("Reward:", reward)
            step += 1
            txt = writeMission(img, obs['mission'])
            txt.save(save_location + '/0.png')
            episode_num += 1
            env.seed(args.seed + episode_num)
            obs = env.reset()
            agent.on_reset()
            step = 1
        else:
            step += 1

    # if renderer.window is None:
    #     break
