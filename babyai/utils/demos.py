import os
import blosc
import pickle
import numpy as np

from .. import utils



def get_demos_path(demos=None, size="10k", env=None, origin=None, valid=False):
    valid_suff = '_valid' if valid else ''
    demos_path = (demos + valid_suff
                  if demos
                  else env + "_" + origin + valid_suff) + '.pkl'
    size = size + '_demos'
    return os.path.join(utils.storage_dir(), size, demos_path)


def load_demos(path, raise_not_found=True):
    path = "." + path
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No demos found at {}".format(path))
        else:
            return []


def save_demos(demos, path):
    utils.create_folders_if_necessary(path)
    pickle.dump(demos, open(path, "wb"))


def synthesize_demos(demos):
    print('{} demonstrations saved'.format(len(demos)))
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    if len(demos) > 0:
        print('Demo num frames: {}'.format(num_frames_per_episode))


def transform_demos(demos):
    '''
    takes as input a list of demonstrations in the format generated with `make_agent_demos` or `make_human_demos`
    i.e. each demo is a tuple (mission, blosc.pack_array(np.array(images)), directions, actions)
    returns demos as a list of lists. Each demo is a list of (obs, action, done) tuples
    '''
    new_demos = []
    for demo in demos:
        new_demo = []

        mission = demo[0]
        all_images = demo[1]
        directions = demo[2]
        actions = demo[3]

        all_images = blosc.unpack_array(all_images)
        n_observations = all_images.shape[0]
        # assert len(directions) == len(actions) == n_observations, "error transforming demos"
        assert len(actions) == n_observations, "error transforming demos"

        for i in range(n_observations):
            if directions is None:
                obs = {'image': all_images[i],
                       'mission': mission}
            else:
                obs = {'image': all_images[i],
                        'direction': directions[i],
                        'mission': mission}
            action = 0 if actions[i] == 6 else actions[i]
            done = i == n_observations - 1
            new_demo.append((obs, action, done))
        new_demos.append(new_demo)
    return new_demos

def transform_merge_demos(demos):
    '''
    takes as input a list of demonstrations in the format generated with `make_agent_demos` or `make_human_demos`
    i.e. each demo is a tuple (mission, blosc.pack_array(np.array(images)), directions, actions)
    returns demos as a list of lists. Each demo is a list of (obs, action, done) tuples
    '''
    new_demos = []
    conjs = [' and ', ' then, ', ' after you ']
    for idx in range(len(demos)//2):
        demo_1 = demos[2 * idx]
        demo_2 = demos[2 * idx + 1]
        conj = conjs[np.random.randint(0, 3)]
        new_demo = []

        if conj == ' after you ':
            mission = demo_2[0] + conj + demo_1[0]
        else:
            mission = demo_1[0] + conj + demo_2[0]

        directions = demo_1[2] + demo_2[2]
        actions = demo_1[3] + demo_2[3]

        all_images = np.concatenate((blosc.unpack_array(demo_1[1]), blosc.unpack_array(demo_2[1])), axis=0)
        n_observations = all_images.shape[0]
        assert len(directions) == len(actions) == n_observations, "error transforming demos"
        for i in range(n_observations):
            obs = {'image': all_images[i],
                   'direction': directions[i],
                   'mission': mission,
                   'submissions': [demo_1[0], demo_2[0]]
                   }
            action = 0 if actions[i] == 6 else actions[i]
            done = i == n_observations - 1
            new_demo.append((obs, action, done))
        new_demos.append(new_demo)
    return new_demos
