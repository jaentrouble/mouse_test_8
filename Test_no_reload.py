import gym
import gym_mouse
import time
import numpy as np
from Agent import Player
import agent_assets.A_hparameters as hp
from tqdm import tqdm
import argparse
import os
import sys
from tensorflow.profiler.experimental import Profile
from datetime import timedelta

ENVIRONMENT = 'mouseCl-v2'

env_kwargs = dict(
    apple_num=10,
    eat_apple = 1.0,
    hit_wall = 0,
)

hp.epsilon = 1
hp.epsilon_min = 0.1
hp.epsilon_nstep = 1500000
hp.Model_save = 200000

parser = argparse.ArgumentParser()
parser.add_argument('-r','--render', dest='render',action='store_true', default=False)
# parser.add_argument('-l', dest='load',default=False)
parser.add_argument('--step', dest='total_steps',default=100000)
# parser.add_argument('--loop', dest='total_loop',default=20)
# parser.add_argument('--curloop', dest='cur_loop',default=0)
parser.add_argument('-n','--logname', dest='log_name',default=False)
# parser.add_argument('--curround', dest='cur_r',default=0)
# parser.add_argument('-lb', dest='load_buffer',action='store_true',default=False)
parser.add_argument('-pf', dest='profile',action='store_true',default=False)
args = parser.parse_args()

vid_type = 'mp4'
total_steps = int(args.total_steps)
# total_loop = int(args.total_loop)
# cur_loop = int(args.cur_loop)
# cur_r = int(args.cur_r)
# load_buffer = args.load_buffer

my_tqdm = tqdm(total=total_steps, dynamic_ncols=True)

if args.render :
    from gym.envs.classic_control.rendering import SimpleImageViewer
    eye_viewer = SimpleImageViewer(maxwidth=1500)
    bar = np.ones((5,3),dtype=np.uint8)*np.array([255,255,0],dtype=np.uint8)
# For benchmark
st = time.time()

env = gym.make(ENVIRONMENT, **env_kwargs)
bef_o = env.reset()

# if args.load :
#     player = Player(env.observation_space, env.action_space, my_tqdm,
#                 args.load, args.log_name, cur_loop*total_steps, cur_r, load_buffer)
if args.log_name:
    # If log directory is explicitely selected
    player = Player(env.observation_space, env.action_space, my_tqdm, 
                log_name=args.log_name)
else :
    player = Player(env.observation_space, env.action_space, my_tqdm)
if args.render :
    env.render()

if args.profile:
    # Warm up
    for step in range(hp.Learn_start+20):
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

    with Profile(f'log/{args.log_name}'):
        for step in range(5):
            action = player.act(bef_o)
            aft_o,r,d,i = env.step(action)
            player.step(bef_o,action,r,d,i)
            if d :
                bef_o = env.reset()
            else:
                bef_o = aft_o
            if args.render :
                env.render()
    remaining_steps = total_steps - hp.Learn_start - 25
    for step in range(remaining_steps):
        if ((hp.Learn_start + 25 + step) % hp.Model_save) == 0 :
            player.save_model()
            score = player.evaluate(gym.make(ENVIRONMENT, **env_kwargs), vid_type)
            print('eval_score:{0}'.format(score))
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

else :
    for step in range(total_steps):
        if (step>0) and ((step % hp.Model_save) == 0) :
            player.save_model()
            score = player.evaluate(gym.make(ENVIRONMENT, **env_kwargs), vid_type)
            print('eval_score:{0}'.format(score))
        action = player.act(bef_o)
        aft_o,r,d,i = env.step(action)
        player.step(bef_o,action,r,d,i)
        if d :
            bef_o = env.reset()
        else:
            bef_o = aft_o
        if args.render :
            env.render()

d = timedelta(seconds=time.time() - st)
print(f'{total_steps}steps took {d}')
my_tqdm.close()

# next_save = player.save_model()
# if not args.load:
#     save_dir = player.save_dir
# else:
#     save_dir, _ = os.path.split(args.load)
# next_dir = os.path.join(save_dir,str(next_save))

# total_loop -= 1
# if total_loop <= 0 :
#     sys.exit()
# else :
#     next_args = []
#     next_args.append('python')
#     next_args.append(__file__)
#     next_args.append('-l')
#     next_args.append(next_dir)
#     next_args.append('--step')
#     next_args.append(str(total_steps))
#     next_args.append('--loop')
#     next_args.append(str(total_loop))
#     next_args.append('--curloop')
#     next_args.append(str(cur_loop+1))
#     next_args.append('--logname')
#     next_args.append(player.log_name)
#     next_args.append('--curround')
#     next_args.append(str(player.rounds))
#     next_args.append('-lb')
    
#     os.execv(sys.executable, next_args)