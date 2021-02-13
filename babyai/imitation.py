import copy
import gym
import time
import datetime
import numpy as np
import sys
import itertools
import torch
import torch.nn.functional as F
from babyai.evaluate import batch_evaluate
import babyai.utils as utils
from babyai.rl import DictList
from babyai.model import ACModel
import multiprocessing
import os
import json
import logging

logger = logging.getLogger(__name__)

import numpy


class EpochIndexSampler:
    """
    Generate smart indices for epochs that are smaller than the dataset size.

    The usecase: you have a code that has a strongly baken in notion of an epoch,
    e.g. you can only validate in the end of the epoch. That ties a lot of
    aspects of training to the size of the dataset. You may want to validate
    more often than once per a complete pass over the dataset.

    This class helps you by generating a sequence of smaller epochs that
    use different subsets of the dataset, as long as this is possible.
    This allows you to keep the small advantage that sampling without replacement
    provides, but also enjoy smaller epochs.
    """
    def __init__(self, n_examples, epoch_n_examples):
        self.n_examples = n_examples
        self.epoch_n_examples = epoch_n_examples

        self._last_seed = None

    def _reseed_indices_if_needed(self, seed):
        if seed == self._last_seed:
            return

        rng = numpy.random.RandomState(seed)
        self._indices = list(range(self.n_examples))
        rng.shuffle(self._indices)
        logger.info('reshuffle the dataset')

        self._last_seed = seed

    def get_epoch_indices(self, epoch):
        """Return indices corresponding to a particular epoch.

        Tip: if you call this function with consecutive epoch numbers,
        you will avoid expensive reshuffling of the index list.

        """
        seed = epoch * self.epoch_n_examples // self.n_examples
        offset = epoch * self.epoch_n_examples % self.n_examples

        indices = []
        while len(indices) < self.epoch_n_examples:
            self._reseed_indices_if_needed(seed)
            n_lacking = self.epoch_n_examples - len(indices)
            indices += self._indices[offset:offset + min(n_lacking, self.n_examples - offset)]
            offset = 0
            seed += 1

        return indices


class ImitationLearning(object):
    def __init__(self, args, ):
        self.args = args

        utils.seed(self.args.seed)
        self.val_seed = self.args.val_seed

        # args.env is a list when training on multiple environments
        if getattr(args, 'multi_env', None):
            self.env = [gym.make(item) for item in args.multi_env]

            self.train_demos = []
            for demos, episodes in zip(args.multi_demos, args.multi_episodes):
                demos_path = utils.get_demos_path(demos, size=self.args.demos_size, valid=False)

                logger.info('loading {} of {} demos'.format(episodes, demos))
                train_demos = utils.load_demos(demos_path)
                logger.info('loaded demos')
                if episodes > len(train_demos):
                    raise ValueError("there are only {} train demos in {}".format(len(train_demos), demos))
                self.train_demos.extend(train_demos[:episodes])
                logger.info('So far, {} demos loaded'.format(len(self.train_demos)))

            self.val_demos = []
            for demos, episodes in zip(args.multi_demos, [args.val_episodes] * len(args.multi_demos)):
                demos_path_valid = utils.get_demos_path(demos, size=self.args.demos_size, valid=True)
                logger.info('loading {} of {} valid demos'.format(episodes, demos))
                valid_demos = utils.load_demos(demos_path_valid)
                logger.info('loaded demos')
                if episodes > len(valid_demos):
                    logger.info('Using all the available {} demos to evaluate valid. accuracy'.format(len(valid_demos)))
                self.val_demos.extend(valid_demos[:episodes])
                logger.info('So far, {} valid demos loaded'.format(len(self.val_demos)))

            logger.info('Loaded all demos')

            observation_space = self.env[0].observation_space
            action_space = self.env[0].action_space

        else:
            self.env = gym.make(self.args.env)

            demos_path = utils.get_demos_path(args.demos, self.args.demos_size, args.env, args.demos_origin, valid=False)
            demos_path_valid = utils.get_demos_path(args.demos, self.args.demos_size, args.env, args.demos_origin, valid=True)

            logger.info('loading demos')
            self.train_demos = utils.load_demos(demos_path)
            logger.info('loaded demos')
            if args.episodes:
                if args.episodes > len(self.train_demos):
                    raise ValueError("there are only {} train demos".format(len(self.train_demos)))
                self.train_demos = self.train_demos[:args.episodes]

            self.val_demos = utils.load_demos(demos_path_valid)
            if args.val_episodes > len(self.val_demos):
                logger.info('Using all the available {} demos to evaluate valid. accuracy'.format(len(self.val_demos)))
            self.val_demos = self.val_demos[:self.args.val_episodes]

            observation_space = self.env.observation_space
            action_space = self.env.action_space

        self.obss_preprocessor = utils.ObssPreprocessor(args.model, observation_space,
                                                        getattr(self.args, 'pretrained_model', None), crafting=self.args.crafting)

        # Define actor-critic model
        model = utils.load_model(args.model, raise_not_found=False)
        if model is None:
            if getattr(self.args, 'pretrained_model', None):
                model = utils.load_model(args.pretrained_model, raise_not_found=True)
                self.acmodel = model['model']
                self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.args.lr, eps=self.args.optim_eps)
                if self.optimizer:
                    self.optimizer.load_state_dict(model['optimizer'])
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
                if self.scheduler:
                    self.scheduler.load_state_dict(model['scheduler'])
            else:

                logger.info('Creating new model')
                in_channels = 3
                if self.args.crafting:
                    in_channels = 9
                self.acmodel = ACModel(self.obss_preprocessor.obs_space, action_space,
                                       args.image_dim, args.memory_dim, args.instr_dim,
                                       not self.args.no_instr, self.args.instr_arch,
                                       not self.args.no_mem, self.args.arch, cpv=self.args.cpv, obs=self.args.obs,
                                       in_channels=in_channels, crafting=self.args.crafting)
                self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.args.lr, eps=self.args.optim_eps)
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        else:
            self.acmodel = model['model']
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.args.lr, eps=self.args.optim_eps)
            if self.optimizer:
                self.optimizer.load_state_dict(model['optimizer'])
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
            if self.scheduler:
                self.scheduler.load_state_dict(model['scheduler'])

        self.obss_preprocessor.vocab.save()
        utils.save_model(self.acmodel, args.model)
        utils.save_optimizer(self.optimizer, self.scheduler, args.model)

        self.acmodel.train()
        if torch.cuda.is_available():
            self.acmodel.cuda()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def default_model_name(args):
        if getattr(args, 'multi_env', None):
            # It's better to specify one's own model name for this scenario
            named_envs = '-'.join(args.multi_env)
        else:
            named_envs = args.env

        # Define model name
        suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        instr = args.instr_arch if args.instr_arch else "noinstr"
        model_name_parts = {
            'envs': named_envs,
            'arch': args.arch,
            'instr': instr,
            'seed': args.seed,
            'suffix': suffix}
        default_model_name = "{envs}_IL_{arch}_{instr}_seed{seed}_{suffix}".format(**model_name_parts)
        if getattr(args, 'pretrained_model', None):
            default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
        return default_model_name

    def starting_indexes(self, num_frames):
        if num_frames % self.args.recurrence == 0:
            return np.arange(0, num_frames, self.args.recurrence)
        else:
            return np.arange(0, num_frames, self.args.recurrence)[:-1]
        
    def get_next_indexes(self, step, demo_starts, demo_lens):
        # for each demo, split into recurrence sized chunks
        # Find which chuncks are still running 
        if step == 0:
            demo_chunk_starts = []
            demo_chunk_lens = []
            full_demo_starts = []
            chunk_num = 0
            for s, l in zip(demo_starts, demo_lens):
                demo_chunk_starts.append(s)
                full_demo_starts.append(chunk_num)
                len_left = l
                num_recs = 0
                chunk_num += 1
                while len_left > self.args.recurrence:
                    num_recs += 1
                    demo_chunk_lens.append(self.args.recurrence)
                    len_left -= self.args.recurrence
                    demo_chunk_starts.append(s + self.args.recurrence * num_recs)
                    chunk_num += 1
                demo_chunk_lens.append(len_left)
            self._demo_chunk_lens = np.array(demo_chunk_lens)
            self._demo_chunk_starts = np.array(demo_chunk_starts)
            self._full_demo_starts = np.array(full_demo_starts)
        starts_to_get = self._demo_chunk_starts[self._demo_chunk_lens > step]
        indexes = starts_to_get + step
        memory_indexes = []
        for t in range(len(self._demo_chunk_starts)):
            if self._demo_chunk_starts[t] in starts_to_get:
                memory_indexes.append(t)
        return indexes, memory_indexes
#         if num_frames % self.args.recurrence == 0:
#             return np.arange(0, num_frames, self.args.recurrence)
#         else:
#             return np.arange(0, num_frames, self.args.recurrence)[:-1]

    def run_epoch_recurrence(self, demos, is_training=False, indices=None):
        if not indices:
            indices = list(range(len(demos)))
            if is_training:
                np.random.shuffle(indices)
        if is_training:
            batch_size = min(self.args.batch_size, len(demos))
        else:
            batch_size = min(self.args.val_batch_size, len(demos))

        if self.args.homomorphic_loss:
            batch_size = batch_size // 2
        offset = 0

        if not is_training:
            self.acmodel.eval()

        # Log dictionary
        log = {"entropy": [], "policy_loss": [], "accuracy": []}

        start_time = time.time()
        frames = 0
        for batch_index in range(len(indices) // batch_size):
            logger.info("batch {}, FPS so far {}".format(
                batch_index, frames / (time.time() - start_time) if frames else 0))
            batch = [demos[i] for i in indices[offset: offset + batch_size]]
            frames += sum([len(demo[3]) for demo in batch])

            _log = self.run_epoch_recurrence_one_batch(batch, is_training=is_training)

            log["entropy"].append(_log["entropy"])
            log["policy_loss"].append(_log["policy_loss"])
            log["accuracy"].append(_log["accuracy"])

            if self.args.complex_batch:
                #_log =  self.run_epoch_recurrence_one_batch(batch, is_training=is_training, complex_batch=True)

                log["complex_entropy"].append(_log["complex_entropy"])
                log["complex_policy_loss"].append(_log["complex_policy_loss"])
                log["complex_accuracy"].append(_log["complex_accuracy"])


            offset += batch_size
        log['total_frames'] = frames

        if not is_training:
            self.acmodel.train()

        return log

    def run_epoch_recurrence_one_batch(self, batch, is_training=False):
        print("one batch")
        reg_batch = utils.demos.transform_demos(batch)
        outputs = self._run_batch(reg_batch, is_training, complex_batch=False)
        final_loss = outputs['final_loss']


        if self.args.homomorphic_loss:
            final_loss += F.mse_loss(outputs['instr_embedding'], outputs['img_embeddings'])
            
        if self.args.complex_batch:
            complex_batch = utils.demos.transform_merge_demos(batch)
            complex_outputs = self._run_batch(complex_batch, is_training, complex_batch=True)
            final_loss += complex_outputs['final_loss']
            if self.args.homomorphic_loss:
                #final_loss += F.mse_loss(complex_outputs['instr_embedding'], complex_outputs['img_embeddings'])
                final_loss += F.mse_loss(complex_outputs['instr_embedding'], complex_outputs['sub_instr_embeddings'])
                #import pdb; pdb.set_trace()
                #final_loss += F.mse_loss(complex_outputs['instr_embedding'], complex_outputs['final_memory_per_demo'])
                #final_loss += F.mse_loss(complex_outputs['final_memory_per_demo'], complex_outputs['summed_memory_per_demo'])
                #final_loss += F.mse_loss(complex_outputs['sub_instr_embeddings'], complex_outputs['summed_memory_per_demo'])
                #import pdb; pdb.set_trace()
        if is_training:
            self.optimizer.zero_grad()
            final_loss.backward()
            self.optimizer.step()

        log = {}
        log["entropy"] = float(outputs['final_entropy'])
        log["policy_loss"] = float(outputs['final_policy_loss'])
        log["accuracy"] = float(outputs['accuracy'])

        if self.args.complex_batch:
            log["complex_entropy"] = float(complex_outputs['final_entropy'])
            log["complex_policy_loss"] = float(complex_outputs['final_policy_loss'])
            log["complex_accuracy"] = float(complex_outputs['accuracy'])
        return log
    
    def _run_batch(self, batch, is_training=False, complex_batch=False):
        batch.sort(key=len, reverse=True)
        # Constructing flat batch and indices pointing to start of each demonstration
        flat_batch = []
        inds = [0]
        demo_starts = []
        demo_ends = []
        for demo in batch:
            demo_starts.append(len(flat_batch))
            flat_batch += demo
            demo_ends.append(len(flat_batch))
            inds.append(inds[-1] + len(demo))
        demo_lens = [end-start for end, start in zip(demo_ends, demo_starts)]
        inds_copy = [i for i in inds]
        flat_batch = np.array(flat_batch)
        inds = inds[:-1]
        num_frames = len(flat_batch)

        mask = np.ones([len(flat_batch)], dtype=np.float64)
        mask[inds] = 0
        mask = torch.tensor(mask, device=self.device, dtype=torch.float).unsqueeze(1)

        # Observations, true action, values and done for each of the stored demostration
        obss, action_true, done = flat_batch[:, 0], flat_batch[:, 1], flat_batch[:, 2]
        action_true = torch.tensor([action for action in action_true], device=self.device, dtype=torch.long)
    
        # Memory to be stored
        memories = torch.zeros([len(flat_batch), self.acmodel.memory_size], device=self.device)
        obs_memories = torch.zeros([len(flat_batch), self.acmodel.memory_size], device=self.device)
        episode_ids = np.zeros(len(flat_batch))
        memory = torch.zeros([len(batch), self.acmodel.memory_size], device=self.device)
        obs_memory = torch.zeros([len(batch), self.acmodel.memory_size], device=self.device)
        preprocessed_first_obs = self.obss_preprocessor(obss[inds], device=self.device, complex=complex_batch)
        instr_embedding = self.acmodel._get_instr_embedding(preprocessed_first_obs.instr)
        sub_instr_embeddings = None
        if complex_batch and self.args.homomorphic_loss:
            sub_instr_embeddings = (self.acmodel._get_instr_embedding(preprocessed_first_obs.subinstr[0]) + self.acmodel._get_instr_embedding(preprocessed_first_obs.subinstr[1]))
        img_embeddings = torch.zeros([len(batch), self.acmodel.semi_memory_size], device=self.device)
        # Loop terminates when every observation in the flat_batch has been handled
        while True:
            # taking observations and done located at inds
            obs = obss[inds]
            done_step = done[inds]
            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)
            with torch.no_grad():
                # taking the memory till len(inds), as demos beyond that have already finished
                out = self.acmodel(
                    preprocessed_obs,
                    memory[:len(inds), :], obs_memory=obs_memory[:len(inds), :], instr_embedding=instr_embedding[:len(inds)])

            new_memory = out['memory']
            memories[inds, :] = memory[:len(inds), :]
            memory[:len(inds), :] = new_memory


            if self.args.obs:
                new_obs_memory = out['extra_predictions']['obs_memory']
                obs_memories[inds, :] = obs_memory[:len(inds), :]
                obs_memory[:len(inds), :] = new_obs_memory

            episode_ids[inds] = range(len(inds))

            # Updating inds, by removing those indices corresponding to which the demonstrations have finished
            inds = inds[:len(inds) - sum(done_step)]
            if self.args.homomorphic_loss and sum(done_step) > 0:
                img_embeddings[len(inds):len(inds) + sum(done_step), :] = out['extra_predictions']['img_embedding'][len(inds):len(inds) + sum(done_step)]


            if len(inds) == 0:
                break

            # Incrementing the remaining indices
            inds = [index + 1 for index in inds]

        # Here, actual backprop upto args.recurrence happens
        #import pdb; pdb.set_trace()
        final_loss = 0
        final_entropy, final_policy_loss, final_value_loss = 0, 0, 0

        #indexes = self.starting_indexes(num_frames)
        indexes, memory_indexes = self.get_next_indexes(step=0, demo_starts=demo_starts, demo_lens=demo_lens)
        memory = memories[indexes]
        obs_memory = obs_memories[indexes]
        accuracy = 0
        total_frames = len(indexes) * self.args.recurrence

        for t in range(self.args.recurrence):
            obs = obss[indexes]
            preprocessed_obs = self.obss_preprocessor(obs, device=self.device)
            action_step = action_true[indexes]
            mask_step = mask[indexes]
#             print("memory shape", memory.shape, "mem index", len(memory_indexes))
#             print("obs_memory shape", obs_memory.shape)
#             print("mask-step", mask_step.shape)
#             if len(memory_indexes) ==  62:
#                 import pdb; pdb.set_trace()
            model_results = self.acmodel(
                preprocessed_obs, memory[memory_indexes] * mask_step, obs_memory = obs_memory[memory_indexes] * mask_step,
                instr_embedding=instr_embedding[episode_ids[indexes]])
            dist = model_results['dist']
            memory[memory_indexes] = model_results['memory']
            if self.args.cpv:
                obs_memory[memory_indexes]  = model_results['extra_predictions']['obs_memory']

            entropy = dist.entropy().mean()
            policy_loss = -dist.log_prob(action_step).mean()

            loss = policy_loss - self.args.entropy_coef * entropy
            action_pred = dist.probs.max(1, keepdim=True)[1]
            accuracy += float((action_pred == action_step.unsqueeze(1)).sum()) / total_frames
            final_loss += loss
            final_entropy += entropy
            final_policy_loss += policy_loss
            indexes, memory_indexes = self.get_next_indexes(step=t+1, demo_starts=demo_starts, demo_lens=demo_lens)

        #import pdb; pdb.set_trace()
        final_loss /= self.args.recurrence
        final_entropy /= self.args.recurrence
        final_policy_loss /= self.args.recurrence
        summed_memory_per_demo = []
        final_memory_per_demo = []
        fl = self._full_demo_starts.tolist()
        demo_idx = 0
        for s,l in zip(fl, fl[1:] + [-1]):
            if l == -1:
                mem_chunks = memory[s:]
            else:
                mem_chunks = memory[s:l]            
            final_memory_per_demo.append(mem_chunks[-1].unsqueeze(0))
            mem_diffs = [mem_chunks[0]]
            total_mem = mem_chunks[0]
            for t in range(1, l-s):
                mem_diffs.append(mem_chunks[t] - total_mem)
                total_mem += mem_diffs[-1]
            summed_memory_per_demo.append(sum(mem_diffs).unsqueeze(0))
        summed_memory_per_demo = torch.cat(summed_memory_per_demo, axis=0)
        final_memory_per_demo = torch.cat(final_memory_per_demo, axis=0)
        import pdb; pdb.set_trace()
        
        return {'final_loss': final_loss,
                'memory': memory,
                'final_memory_per_demo': final_memory_per_demo,
                'summed_memory_per_demo': summed_memory_per_demo,
                'obs_memory': obs_memory,
                'instr_embedding': instr_embedding,
                'sub_instr_embeddings': sub_instr_embeddings,
                'img_embeddings' : img_embeddings,
                'final_entropy': final_entropy,
                'final_policy_loss': final_policy_loss,
                'accuracy': accuracy,
               }

    def validate(self, episodes, verbose=True):
        if verbose:
            logger.info("Validating the model")
        if getattr(self.args, 'multi_env', None):
            agent = utils.load_agent(self.env[0], model_name=self.args.model, argmax=True)
        else:
            agent = utils.load_agent(self.env, model_name=self.args.model, argmax=True)

        # Setting the agent model to the current model
        agent.model = self.acmodel

        agent.model.eval()
        logs = []

        for env_name in ([self.args.env] if not getattr(self.args, 'multi_env', None)
                         else self.args.multi_env):
            logs += [batch_evaluate(agent, env_name, self.val_seed, episodes)]
            self.val_seed += episodes
        agent.model.train()

        return logs

    def collect_returns(self):
        logs = self.validate(episodes=self.args.eval_episodes, verbose=False)
        mean_return = {tid: np.mean(log["return_per_episode"]) for tid, log in enumerate(logs)}
        return mean_return

    def train(self, train_demos, writer, csv_writer, status_path, header, reset_status=False):
        # Load the status
        print("TRAINING")
        def initial_status():
            return {'i': 0,
                    'num_frames': 0,
                    'patience': 0}

        status = initial_status()
        if os.path.exists(status_path) and not reset_status:
            with open(status_path, 'r') as src:
                status = json.load(src)
        elif not os.path.exists(os.path.dirname(status_path)):
            # Ensure that the status directory exists
            os.makedirs(os.path.dirname(status_path))

        # If the batch size is larger than the number of demos, we need to lower the batch size
        if self.args.batch_size > len(train_demos):
            self.args.batch_size = len(train_demos)
            logger.info("Batch size too high. Setting it to the number of train demos ({})".format(len(train_demos)))

        # Model saved initially to avoid "Model not found Exception" during first validation step
        utils.save_model(self.acmodel, self.args.model)

        # best mean return to keep track of performance on validation set
        best_success_rate, patience, i = 0, 0, 0
        total_start_time = time.time()

        epoch_length = self.args.epoch_length
        if not epoch_length:
            epoch_length = len(train_demos)
        index_sampler = EpochIndexSampler(len(train_demos), epoch_length)
        print("HELLO", status['i'] , getattr(self.args, 'epochs', int(1e9)))
        while status['i'] < getattr(self.args, 'epochs', int(1e9)):
            print("HELLO", status['i'] , "<", getattr(self.args, 'epochs', int(1e9)), status['i'] < getattr(self.args, 'epochs', int(1e9)))
            if 'patience' not in status:  # if for some reason you're finetuining with IL an RL pretrained agent
                status['patience'] = 0
            # Do not learn if using a pre-trained model that already lost patience
            if status['patience'] > self.args.patience:
                print("status['patience'] > self.args.patience", status['patience'] > self.args.patience)
                break
            if status['num_frames'] > self.args.frames:
                print("status['num_frames'] > self.args.frames", status['num_frames'] > self.args.frames)
                break

            update_start_time = time.time()

            indices = index_sampler.get_epoch_indices(status['i'])
            print("gonna run a batch")
            log = self.run_epoch_recurrence(train_demos, is_training=True, indices=indices)

            # Learning rate scheduler
            self.scheduler.step()

            status['num_frames'] += log['total_frames']
            status['i'] += 1

            update_end_time = time.time()

            # Print logs
            if status['i'] % self.args.log_interval == 0:
                total_ellapsed_time = int(time.time() - total_start_time)

                fps = log['total_frames'] / (update_end_time - update_start_time)
                duration = datetime.timedelta(seconds=total_ellapsed_time)

                for key in log:
                    log[key] = np.mean(log[key])

                train_data = [status['i'], status['num_frames'], fps, total_ellapsed_time,
                              log["entropy"], log["policy_loss"], log["accuracy"]]

                logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | H {:.3f} | pL {: .3f} | A {: .3f}".format(*train_data))

                # Log the gathered data only when we don't evaluate the validation metrics. It will be logged anyways
                # afterwards when status['i'] % self.args.val_interval == 0
                if status['i'] % self.args.val_interval != 0:
                    # instantiate a validation_log with empty strings when no validation is done
                    validation_data = [''] * len([key for key in header if 'valid' in key])
                    assert len(header) == len(train_data + validation_data)
                    if self.args.tb:
                        for key, value in zip(header, train_data):
                            writer.add_scalar(key, float(value), status['num_frames'])
                    csv_writer.writerow(train_data + validation_data)

            if status['i'] % self.args.val_interval == 0:

                valid_log = self.validate(self.args.val_episodes)
                mean_return = [np.mean(log['return_per_episode']) for log in valid_log]
                success_rate = [np.mean([1 if r > 0 else 0 for r in log['return_per_episode']]) for log in
                                valid_log]

                val_log = self.run_epoch_recurrence(self.val_demos)
                validation_accuracy = np.mean(val_log["accuracy"])

                if status['i'] % self.args.log_interval == 0:
                    validation_data = [validation_accuracy] + mean_return + success_rate
                    logger.info(("Validation: A {: .3f} " + ("| R {: .3f} " * len(mean_return) +
                                                             "| S {: .3f} " * len(success_rate))
                                 ).format(*validation_data))

                    assert len(header) == len(train_data + validation_data)
                    if self.args.tb:
                        for key, value in zip(header, train_data + validation_data):
                            writer.add_scalar(key, float(value), status['num_frames'])
                    csv_writer.writerow(train_data + validation_data)

                # In case of a multi-env, the update condition would be "better mean success rate" !
                if np.mean(success_rate) > best_success_rate:
                    best_success_rate = np.mean(success_rate)
                    status['patience'] = 0
                    with open(status_path, 'w') as dst:
                        json.dump(status, dst)
                    # Saving the model
                    logger.info("Saving best model")

                    if torch.cuda.is_available():
                        self.acmodel.cpu()
                    utils.save_model(self.acmodel, self.args.model + "_best")
                    self.obss_preprocessor.vocab.save(utils.get_vocab_path(self.args.model + "_best"))
                    if torch.cuda.is_available():
                        self.acmodel.cuda()
                else:
                    status['patience'] += 1
                    logger.info(
                        "Losing patience, new value={}, limit={}".format(status['patience'], self.args.patience))

                if torch.cuda.is_available():
                    self.acmodel.cpu()
                utils.save_model(self.acmodel, self.args.model)
                utils.save_optimizer(self.optimizer, self.scheduler, self.args.model)
                utils.save_model(self.acmodel, self.args.model + "/" + str(status['i']))


                self.obss_preprocessor.vocab.save()
                if torch.cuda.is_available():
                    self.acmodel.cuda()
                with open(status_path, 'w') as dst:
                    json.dump(status, dst)

        return best_success_rate
