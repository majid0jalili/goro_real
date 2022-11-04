import torch
import collections
from random import random
from random import randint
import numpy as np
from typing import Dict, List, Tuple
from segment_tree import MinSegmentTree, SumSegmentTree


class ReplayBuffer():
    def __init__(self, buffer_limit, action_space, device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.action_space = action_space
        self.buffer_limit = buffer_limit
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)

    def clear_buf(self):
        self.buffer = collections.deque(maxlen=self.buffer_limit)

    # write the buffer to a csv file
    def write_to_csv(self, filename):
        import csv
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['state', 'action', 'reward', 'next_state', 'done']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for transition in self.buffer:
                state, actions, reward, next_state, done_mask = transition
                writer.writerow({'state': state, 'action': actions, 'reward': reward,
                                 'next_state': next_state, 'done': done_mask})

    def load_from_csv(self, filename):
        import csv
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                state = row['state']
                actions = row['action']
                reward = row['reward']
                next_state = row['next_state']
                done_mask = row['done']
                self.put((state, actions, reward, next_state, done_mask))

    def write_buffer(self, state, next_state, actions, reward):
        state = np.array(state)
        next_state = np.array(next_state)
        actions = np.array(actions)
        reward = np.array(reward)[0]
        # print("Writing to buffer", state, actions, reward)
        self.put((state, actions, reward, next_state, 0))

    def print_buffer(self):
        print(self.buffer)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_lst, reward_lst, next_state_lst, done_mask_lst, actions_lst = [], [], [], [], []

        actions_lst = [[] for i in range(self.action_space)]
        for transition in mini_batch:
            state, actions, reward, next_state, done_mask = transition
            state_lst.append(state)
            for idx in range(self.action_space):
                actions_lst[idx].append(actions[idx])
            reward_lst.append([reward])
            next_state_lst.append(next_state)
            done_mask_lst.append([done_mask])
        actions_lst = [torch.tensor(x, dtype=torch.float).to(
            self.device) for x in actions_lst]

        return torch.tensor(state_lst, dtype=torch.float).to(self.device),\
            actions_lst,\
            torch.tensor(reward_lst, dtype=torch.float).to(self.device),\
            torch.tensor(next_state_lst, dtype=torch.float).to(self.device),\
            torch.tensor(done_mask_lst).to(self.device),\
            [], []

    def __len__(self):
        return len(self.buffer)
        
    def size(self):
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(self, buffer_limit, action_space, device, alpha: float = 0.6, beta: float = 0.4):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(
            buffer_limit, action_space, device)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.beta = beta

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.buffer_limit:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def write_buffer(self, state, next_state, actions, reward):
        super().write_buffer(state, next_state, actions, reward)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.buffer_limit

    def sample(self, batch_size):
        assert len(self) >= batch_size
        assert self.beta > 0

        indices = self._sample_proportional(batch_size)
        res_list = map(self.buffer.__getitem__, indices)
        mini_batch = list(res_list)
        toss = random()
        if (toss < 0.1):
            mini_batch = random.sample(self.buffer, n)
            
        state_lst, reward_lst, next_state_lst, done_mask_lst, actions_lst = [], [], [], [], []

        actions_lst = [[] for i in range(self.action_space)]
        for transition in mini_batch:
            state, actions, reward, next_state, done_mask = transition
            state_lst.append(state)
            for idx in range(self.action_space):
                actions_lst[idx].append(actions[idx])
            reward_lst.append([reward])
            next_state_lst.append(next_state)
            done_mask_lst.append([done_mask])
        actions_lst = [torch.tensor(x, dtype=torch.float).to(
            self.device) for x in actions_lst]

        weights = np.array([self._calculate_weight(i, self.beta)
                           for i in indices])

        return torch.tensor(state_lst, dtype=torch.float).to(self.device),\
            actions_lst,\
            torch.tensor(reward_lst, dtype=torch.float).to(self.device),\
            torch.tensor(next_state_lst, dtype=torch.float).to(self.device),\
            torch.tensor(done_mask_lst).to(self.device),\
            torch.tensor(weights, dtype=torch.float).to(self.device),\
            indices

    def _sample_proportional(self, batch_size) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size
        print("p_total, segment", p_total, segment)
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            print(a, b, upperbound, idx)
            indices.append(idx)

        return indices

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight
