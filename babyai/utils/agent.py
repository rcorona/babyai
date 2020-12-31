from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .. import utils
from babyai.bot import Bot
from babyai.model import ACModel
from random import Random
from argparse import Namespace
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import pdb
# import revtok

class Agent(ABC):
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def on_reset(self):
        pass

    @abstractmethod
    def act(self, obs):
        """Propose an action based on observation.

        Returns a dict, with 'action` entry containing the proposed action,
        and optionaly other entries containing auxiliary information
        (e.g. value function).

        """
        pass

    @abstractmethod
    def analyze_feedback(self, reward, done):
        pass



class CPVPolicy(nn.Module):

    def __init__(self, model_path):
        super().__init__()

        self.batch_size = 256

        # Unpack model dict.
        model_dict = torch.load(model_path)
        self.args = model_dict['args']
        self.vocab = model_dict['vocab']
        self.device = torch.device('cuda') if self.args.gpu else torch.device('cpu')

        print(self.vocab)
        print([self.vocab.index2word(i) for i in range(35)])


        self.img_shape = 7 * 7 * (11 + 6 + 3) # This is based off of the Babyai img size
        self.num_actions = 7

        self.embed = nn.Embedding(len(self.vocab), self.args.demb)
        self.linear = nn.Linear(self.img_shape, self.args.demb)
        self.lang_enc = nn.LSTM(self.args.demb, self.args.dhid, bidirectional=False, num_layers=2, batch_first=True)
        self.img_enc = nn.LSTM(self.args.demb, self.args.dhid, bidirectional=False, num_layers=2, batch_first=True)
        self.obs_enc = nn.LSTM(self.args.demb, self.args.dhid, bidirectional=False, num_layers=2, batch_first=True)

        if hasattr(self.args, 'baseline') and self.args.baseline:
            self.im_linear_1 =  nn.Linear(self.args.dhid * 3, self.args.demb)
        else:
            self.im_linear_1 =  nn.Linear(self.args.dhid * 2, self.args.demb)

        self.im_linear_2 =  nn.Linear(self.args.demb, self.num_actions)

        # Load pre-trained model.
        self.load_state_dict(model_dict['model'])
        self.to(self.device)

        self.reset()

    def reset(self):
        # Hidden states for LSTMs.
        self.context_h = torch.zeros(2, self.batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
        self.context_c = torch.zeros(2, self.batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H

        self.obs_h = torch.zeros(2, self.batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
        self.obs_c = torch.zeros(2, self.batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H

    def language_encoder(self, batch, batch_size, h_0=None, c_0=None):
        '''
        Encodes a stacked tensor.
        '''

        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
            c_0 = torch.zeros(2, batch_size, self.args.dhid).type(torch.float).to(self.device) # -> 2 x B x H
        out, (h, c) = self.lang_enc(batch, (h_0, c_0)) # -> 2 x B x H

        hid_sum = h[-1].squeeze() # -> B x H

        return hid_sum, h, c

    def act(self, imgs, high, high_lens):
        '''

        '''
        B = imgs.shape[0]

        # Put on device.
        imgs = imgs.to(self.device)
        high = high.to(self.device)
        high_lens = high_lens.to(self.device)

        # Embed image observation.
        imgs = self.linear(imgs)
        context, (self.context_h, self.context_c) = self.img_enc(imgs, (self.context_h, self.context_c))
        obs, (self.obs_h, self.obs_c) = self.obs_enc(imgs, (self.obs_h, self.obs_c))

        context = context.squeeze()
        obs = obs.squeeze()

        # Embed language.
        high = self.embed(high) # -> B x M x D
        high = pack_padded_sequence(high, high_lens, batch_first=True, enforce_sorted=False)
        high, _, _ = self.language_encoder(high, B) # -> B x H

        ## Pass through policy.

        if hasattr(self.args, 'baseline') and self.args.baseline:
            state = torch.cat([high, context, obs], dim=1)
        else:
            ## plan -> high - context
            plan = high - context

            ## state -> concat plan and current
            state = torch.cat([plan, obs], dim=1)

        ## put state through ff
        state = self.im_linear_1(state)
        state = F.relu(state)
        state = self.im_linear_2(state)

        # Distribution over actions.
        dist = F.softmax(state, dim=1)
        action = torch.argmax(dist, dim=1)

        return {'dist': dist, 'action': action}

class CPVAgent(Agent):

    def __init__(self, model_path):
        self.model = CPVPolicy(model_path)
        self.model.eval()

        # For vectorizing binary features.
        self.object_default = np.array([np.eye(11) for _ in range(49)])
        self.color_default = np.array([np.eye(6) for _ in range(49)])
        self.state_default = np.array([np.eye(3) for _ in range(49)])

        self.final_shape = 7 * 7 * (11 + 6 + 3)

    def reset(self):
        self.model.reset()

    def act_batch(self, many_obs):

        # Unpack and preprocess image observation.
        imgs = [np.reshape(obs['image'], (49, -1)) for obs in many_obs]

        low_level_object = [torch.tensor(self.object_default[list(range(49)), img[:, 0], :], dtype=torch.float) for img in imgs]
        low_level_color = [torch.tensor(self.color_default[list(range(49)), img[:, 1], :], dtype=torch.float) for img in imgs]
        low_level_state = [torch.tensor(self.state_default[list(range(49)), img[:, 2], :], dtype=torch.float) for img in imgs]

        imgs = [torch.cat([low_level_object[i], low_level_color[i], low_level_state[i]], dim=1).reshape(self.final_shape) for i in range(len(imgs))]
        imgs = torch.stack(imgs).unsqueeze(1)

        # Preprocess Language
        # highs = [torch.tensor([self.model.vocab.word2index(word.strip()) for word in revtok.tokenize(o['mission'])]) for o in many_obs]
        high_lens = torch.tensor([len(high) for high in highs])
        highs = pad_sequence(highs, batch_first=True)

        with torch.no_grad():
            return self.model.act(imgs, highs, high_lens)

    def act(self, obs):
        return self.act_batch([obs])

    def analyze_feedback(self, reward, done):
        pass

class ModelAgent(Agent):
    """A model-based agent. This agent behaves using a model."""

    def __init__(self, model_or_name, obss_preprocessor, argmax):
        if obss_preprocessor is None:
            assert isinstance(model_or_name, str)
            obss_preprocessor = utils.ObssPreprocessor(model_or_name)
        self.obss_preprocessor = obss_preprocessor
        if isinstance(model_or_name, str):
            self.model = utils.load_model(model_or_name)
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            self.model = model_or_name
        self.device = next(self.model.parameters()).device
        self.argmax = argmax
        self.memory = None

    def act_batch(self, many_obs):
        if self.memory is None:
            self.memory = torch.zeros(
                len(many_obs), self.model.memory_size, device=self.device)
        elif self.memory.shape[0] != len(many_obs):
            raise ValueError("stick to one batch size for the lifetime of an agent")
        preprocessed_obs = self.obss_preprocessor(many_obs, device=self.device)

        with torch.no_grad():
            model_results = self.model(preprocessed_obs, self.memory)
            dist = model_results['dist']
            value = model_results['value']
            self.memory = model_results['memory']

        if self.argmax:
            action = dist.probs.argmax(1)
        else:
            action = dist.sample()

        return {'action': action,
                'dist': dist,
                'value': value}

    def act(self, obs):
        return self.act_batch([obs])

    def analyze_feedback(self, reward, done):
        if isinstance(done, tuple):
            for i in range(len(done)):
                if done[i]:
                    self.memory[i, :] *= 0.
        else:
            self.memory *= (1 - done)


class RandomAgent:
    """A newly initialized model-based agent."""

    def __init__(self, seed=0, number_of_actions=7):
        self.rng = Random(seed)
        self.number_of_actions = number_of_actions

    def act(self, obs):
        action = self.rng.randint(0, self.number_of_actions - 1)
        # To be consistent with how a ModelAgent's output of `act`:
        return {'action': torch.tensor(action),
                'dist': None,
                'value': None}


class DemoAgent(Agent):
    """A demonstration-based agent. This agent behaves using demonstrations."""

    def __init__(self, demos_name, env_name, origin):
        self.demos_path = utils.get_demos_path(demos_name, env_name, origin, valid=False)
        self.demos = utils.load_demos(self.demos_path)
        self.demos = utils.demos.transform_demos(self.demos)
        self.demo_id = 0
        self.step_id = 0

    @staticmethod
    def check_obss_equality(obs1, obs2):
        if not(obs1.keys() == obs2.keys()):
            return False
        for key in obs1.keys():
            if type(obs1[key]) in (str, int):
                if not(obs1[key] == obs2[key]):
                    return False
            else:
                if not (obs1[key] == obs2[key]).all():
                    return False
        return True

    def act(self, obs):
        if self.demo_id >= len(self.demos):
            raise ValueError("No demonstration remaining")
        expected_obs = self.demos[self.demo_id][self.step_id][0]
        assert DemoAgent.check_obss_equality(obs, expected_obs), "The observations do not match"

        return {'action': self.demos[self.demo_id][self.step_id][1]}

    def analyze_feedback(self, reward, done):
        self.step_id += 1

        if done:
            self.demo_id += 1
            self.step_id = 0


class BotAgent:
    def __init__(self, env):
        """An agent based on a GOFAI bot."""
        self.env = env
        self.on_reset()

    def on_reset(self):
        self.bot = Bot(self.env)

    def act(self, obs=None, update_internal_state=True, *args, **kwargs):
        action = self.bot.replan()
        return {'action': action}

    def analyze_feedback(self, reward, done):
        pass


def load_agent(env, model_name, demos_name=None, demos_origin=None, argmax=True, env_name=None, model_path=None):
    # env_name needs to be specified for demo agents
    
    if model_name == 'BOT':
        return BotAgent(env)

    elif model_name == 'cpv':
        return CPVAgent(model_path)

    elif model_name is not None:
        obss_preprocessor = utils.ObssPreprocessor(model_name, env.observation_space)
        return ModelAgent(model_name, obss_preprocessor, argmax)
    elif demos_origin is not None or demos_name is not None:
        return DemoAgent(demos_name=demos_name, env_name=env_name, origin=demos_origin)
