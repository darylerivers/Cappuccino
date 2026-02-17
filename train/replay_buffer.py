import os

import numpy as np
import numpy.random as rd
import torch


class ReplayBuffer:  # for off-policy
    def __init__(self, max_len, state_dim, action_dim, gpu_id=0):
        self.now_len = 0
        self.next_idx = 0
        self.prev_idx = 0
        self.if_full = False
        self.max_len = max_len
        self.action_dim = action_dim
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu"
        )

        other_dim = 1 + 1 + self.action_dim  # reward_dim + mask_dim + action_dim
        self.buf_other = torch.empty(
            (max_len, other_dim), dtype=torch.float32, device=self.device
        )

        buf_state_size = (
            (max_len, state_dim)
            if isinstance(state_dim, int)
            else (max_len, *state_dim)
        )
        self.buf_state = torch.empty(
            buf_state_size, dtype=torch.float32, device=self.device
        )

    def extend_buffer(self, state, other):
        size = len(other)
        next_idx = self.next_idx + size

        if next_idx > self.max_len:
            # Transfer remaining data to the end of the buffer
            remaining_size = self.max_len - self.next_idx
            self.buf_state[self.next_idx : self.max_len].copy_(
                state[:remaining_size], non_blocking=True
            )
            self.buf_other[self.next_idx : self.max_len].copy_(
                other[:remaining_size], non_blocking=True
            )
            self.if_full = True

            # Wrap around to the beginning of the buffer
            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx].copy_(state[-next_idx:], non_blocking=True)
            self.buf_other[0:next_idx].copy_(other[-next_idx:], non_blocking=True)
        else:
            # Direct copy without slicing
            self.buf_state[self.next_idx : next_idx].copy_(
                state, non_blocking=True if self.device.type == 'cuda' else False
            )
            self.buf_other[self.next_idx : next_idx].copy_(
                other, non_blocking=True if self.device.type == 'cuda' else False
            )
        self.next_idx = next_idx


#    def update_buffer(self, traj_lists):
#        r_exp = 0.0
#        for traj_list in traj_lists:
#            self.extend_buffer(state=traj_list[0], other=torch.hstack(traj_list[1:]))
#
#            steps += traj_list[1].shape[0]
#            r_exp += traj_list[1].mean().item()
#        return steps, r_exp / len(traj_lists)

    def sample_batch(self, batch_size) -> tuple:
        indices = rd.randint(self.now_len - 1, size=batch_size)
        # r_m_a = self.buf_other[indices]
        # return (r_m_a[:, 0:1],
        #         r_m_a[:, 1:2],
        #         r_m_a[:, 2:],
        #         self.buf_state[indices],
        #         self.buf_state[indices + 1])
        return (
            self.buf_other[indices, 0:1],
            self.buf_other[indices, 1:2],
            self.buf_other[indices, 2:],
            self.buf_state[indices],
            self.buf_state[indices + 1],
        )

    def sample_batch_r_m_a_s(self):
        if self.prev_idx <= self.next_idx:
            r = self.buf_other[self.prev_idx : self.next_idx, 0:1]
            m = self.buf_other[self.prev_idx : self.next_idx, 1:2]
            a = self.buf_other[self.prev_idx : self.next_idx, 2:]
            s = self.buf_state[self.prev_idx : self.next_idx]
        else:
            r = torch.vstack(
                (
                    self.buf_other[self.prev_idx :, 0:1],
                    self.buf_other[: self.next_idx, 0:1],
                )
            )
            m = torch.vstack(
                (
                    self.buf_other[self.prev_idx :, 1:2],
                    self.buf_other[: self.next_idx, 1:2],
                )
            )
            a = torch.vstack(
                (
                    self.buf_other[self.prev_idx :, 2:],
                    self.buf_other[: self.next_idx, 2:],
                )
            )
            s = torch.vstack(
                (
                    self.buf_state[self.prev_idx :],
                    self.buf_state[: self.next_idx],
                )
            )
        self.prev_idx = self.next_idx
        return r, m, a, s  # reward, mask, action, state

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def save_or_load_history(self, cwd, if_save, buffer_id=0):
        save_path = f"{cwd}/replay_{buffer_id}.npz"

        if if_save:
            self.update_now_len()
            state_dim = self.buf_state.shape[1]
            other_dim = self.buf_other.shape[1]

            # Allocate numpy arrays with the correct shape
            buf_state = np.empty((self.now_len, state_dim), dtype=np.float16)
            buf_other = np.empty((self.now_len, other_dim), dtype=np.float16)

            # Copy data from GPU to CPU (if on GPU)
            if self.device.type == 'cuda':
                buf_state[:] = (
                    self.buf_state[:self.now_len].detach().cpu().numpy()
                )
                buf_other[:] = (
                    self.buf_other[:self.now_len].detach().cpu().numpy()
                )
            else:
                buf_state[:] = self.buf_state[:self.now_len].numpy()
                buf_other[:] = self.buf_other[:self.now_len].numpy()

            np.savez_compressed(save_path, buf_state=buf_state, buf_other=buf_other)

            print(f"| ReplayBuffer save in: {save_path}")
        elif os.path.isfile(save_path):
            buf_dict = np.load(save_path)
            buf_state = buf_dict["buf_state"]
            buf_other = buf_dict["buf_other"]

            buf_state = torch.as_tensor(
                buf_state, dtype=torch.float32, device=self.device
            )
            buf_other = torch.as_tensor(
                buf_other, dtype=torch.float32, device=self.device
            )
            self.extend_buffer(buf_state, buf_other)
            self.update_now_len()
            print(f"| ReplayBuffer load: {save_path}")


class ReplayBufferListHybrid(list):
    """Optimized replay buffer with adaptive strategy."""

    def __init__(self, gpu_id=0, pin_to_gpu=True):
        list.__init__(self)
        self.pin_to_gpu = pin_to_gpu
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0) and pin_to_gpu) else "cpu"
        )
        self._last_size = 0

    def update_buffer(self, traj_list):
        if not traj_list:
            return 0, 0.0

        num_items = len(traj_list[0])
        total_steps = sum(len(traj[0]) for traj in traj_list)

        # Clear old buffer BEFORE allocating new one
        old_buffer = list(self)  # Save reference
        self.clear()
        del old_buffer  # Explicit delete

        # Force immediate cleanup if memory pressure
        if self.pin_to_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Build new buffer efficiently
        result = []
        for item_idx in range(num_items):
            if self.pin_to_gpu and torch.cuda.is_available():
                # Pre-allocate on GPU
                first_item = traj_list[0][item_idx]
                item_shape = [total_steps] + list(first_item.shape[1:])
                combined = torch.empty(item_shape,
                                      dtype=first_item.dtype,
                                      device=self.device)

                # Fill with non-blocking transfers
                offset = 0
                for traj in traj_list:
                    item = traj[item_idx]
                    item_len = len(item)
                    combined[offset:offset+item_len].copy_(item, non_blocking=True)
                    offset += item_len

                result.append(combined)
            else:
                # CPU path: simple concatenation
                items = [traj[item_idx] for traj in traj_list]
                result.append(torch.cat(items, dim=0))
                del items

        # Update buffer
        self.extend(result)

        # Cleanup
        if self.pin_to_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()

        steps = self[1].shape[0]
        r_exp = self[1].mean().item()
        return steps, r_exp


class ReplayBufferMP:
    def __init__(
        self,
        gpu_id,
        max_len,
        state_dim,
        action_dim,
        buffer_num,
        if_use_per,
    ):
        """Experience Replay Buffer for Multiple Processing

        save environment transition in a continuous RAM for high performance training
        we save trajectory in order and save state and other (action, reward, mask, ...) separately.

        :param gpu_id: [int] create buffer space on CPU RAM or GPU, `-1` denotes create on CPU
        :param max_len: [int] the max_len of ReplayBuffer, not the total len of ReplayBufferMP
        :param state_dim: [int] the dimension of state
        :param action_dim: [int] the dimension of action (action_dim==1 for discrete action)
        :param buffer_num: [int] the number of ReplayBuffer in ReplayBufferMP, equal to args.worker_num
        :param if_use_per: [bool] PRE: Prioritized Experience Replay for sparse reward
        """

        """Experience Replay Buffer for Multiple Processing

        `int max_len` 
        `int worker_num` the rollout workers number
        """
        self.now_len = 0
        self.max_len = max_len
        self.worker_num = buffer_num

        buf_max_len = max_len // buffer_num
        self.buffers = [
            ReplayBuffer(
                max_len=buf_max_len,
                state_dim=state_dim,
                action_dim=action_dim,
                if_use_per=if_use_per,
                gpu_id=gpu_id,
            )
            for _ in range(buffer_num)
        ]

    def sample_batch(self, batch_size) -> list:
        bs = batch_size // self.worker_num
        #list_items = [self.buffers[i].sample_batch(bs) for i in range(self.worker_num)]
        # list_items of reward, mask, action, state, next_state
        # list_items of reward, mask, action, state, next_state, is_weights (PER)

        #list_items = list(map(list, zip(*list_items)))  # 2D-list transpose
        #return [torch.cat(item, dim=0) for item in list_items]
        result = []
        for i in range(len(self.buffers[0].sample_batch(bs))):
            stacked = torch.stack([buf.sample_batch(bs)[i] for buf in self.buffers], dim=0)
            if self.device.type == 'cuda':
                stacked = stacked.to(self.device, non_blocking=True)
            result.append(stacked)
        return result

    def sample_batch_one_step(self, batch_size):
        indices = rd.randint(self.now_len - 1, size=batch_size)
        return (
            self.buf_other[indices, 0:1],
            self.buf_other[indices, 1:2],
            self.buf_other[indices, 2:],
            self.buf_state[indices],
            self.buf_state[indices + 1 if indices + 1 < self.now_len else 0],  # Handle wrap-around
        )

        # list_items of reward, mask, action, state, next_state
        # list_items of reward, mask, action, state, next_state, is_weights (PER)

        list_items = list(map(list, zip(*list_items)))  # 2D-list transpose
        return [torch.cat(item, dim=0) for item in list_items]

    def update_now_len(self):
        self.now_len = 0
        for buffer in self.buffers:
            buffer.update_now_len()
            self.now_len += buffer.now_len

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        # for buffer in self.l_buffer:
        self.buffers[0].print_state_norm(neg_avg, div_std)

    def td_error_update(self, td_error):
        td_errors = td_error.view(self.worker_num, -1, 1)
        for i in range(self.worker_num):
            self.buffers[i].per_tree.td_error_update(td_errors[i])

    def save_or_load_history(self, cwd, if_save):
        for i in range(self.worker_num):
            self.buffers[i].save_or_load_history(cwd, if_save, buffer_id=i)


class BinarySearchTree:
    """Binary Search Tree for PER

    Contributor: Github GyChou, Github mississippiu
    Reference: https://github.com/kaixindelele/DRLib/tree/main/algos/pytorch/td3_sp
    Reference: https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    """

    def __init__(self, memo_len):
        self.memo_len = memo_len  # replay buffer len
        self.prob_ary = np.zeros(
            (memo_len - 1) + memo_len
        )  # parent_nodes_num + leaf_nodes_num
        self.max_len = len(self.prob_ary)
        self.now_len = self.memo_len - 1  # pointer
        self.indices = None
        self.depth = int(np.log2(self.max_len))

        # PER.  Prioritized Experience Replay. Section 4
        # alpha, beta = 0.7, 0.5 for rank-based variant
        # alpha, beta = 0.6, 0.4 for proportional variant
        self.per_alpha = 0.6  # alpha = (Uniform:0, Greedy:1)
        self.per_beta = 0.4  # beta = (PER:0, NotPER:1)

    def update_id(self, data_id, prob=10):  # 10 is max_prob
        tree_id = data_id + self.memo_len - 1
        if self.now_len == tree_id:
            self.now_len += 1

        delta = prob - self.prob_ary[tree_id]
        self.prob_ary[tree_id] = prob

        while tree_id != 0:  # propagate the change through tree
            tree_id = (tree_id - 1) // 2  # faster than the recursive loop
            self.prob_ary[tree_id] += delta

    def update_ids(self, data_ids, prob=10):
        # Validate data_ids and convert to internal indices
        ids = np.array(data_ids) + self.memo_len - 1
        if np.any(ids >= len(self.prob_ary)):
            raise ValueError("data_ids out of bounds")

        # Update the number of valid entries
        self.now_len += (ids < self.now_len).sum()

        upper_step = self.depth - 1
        p_ids = (ids - 1) // 2

        while upper_step:
            for pid in p_ids:
                if pid >= len(self.prob_ary):
                    continue
                left_child = 2 * pid + 1
                right_child = left_child + 1
                self.prob_ary[pid] = (
                    self.prob_ary[left_child] + self.prob_ary[right_child]
                )
            p_ids = (p_ids - 1) // 2
            upper_step -= 1

        if len(self.prob_ary) > 0:
            self.prob_ary[0] = self.prob_ary[1] + self.prob_ary[2]


    def get_leaf_id(self, v):
        """Tree structure and array storage:

        Tree index:
              0       -> storing priority sum
            |  |
          1     2
         | |   | |
        3  4  5  6    -> storing priority for transitions
        Array type for storing: [0, 1, 2, 3, 4, 5, 6]
        """
        parent_idx = 0
        while True:
            l_idx = 2 * parent_idx + 1  # the leaf's left node
            r_idx = l_idx + 1  # the leaf's right node
            if l_idx >= (len(self.prob_ary)):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.prob_ary[l_idx]:
                    parent_idx = l_idx
                else:
                    v -= self.prob_ary[l_idx]
                    parent_idx = r_idx
        return min(leaf_idx, self.now_len - 2)  # leaf_idx

    def get_indices_is_weights(self, batch_size, beg, end):
        self.per_beta = min(1.0, self.per_beta + 0.001)

        # get random values for searching indices with proportional prioritization
        values = (rd.rand(batch_size) + np.arange(batch_size)) * (
            self.prob_ary[0] / batch_size
        )

        # get proportional prioritization
        leaf_ids = np.array([self.get_leaf_id(v) for v in values])
        self.indices = leaf_ids - (self.memo_len - 1)

        prob_ary = self.prob_ary[leaf_ids] / self.prob_ary[beg:end].min()
        is_weights = np.power(prob_ary, -self.per_beta)  # important sampling weights
        return self.indices, is_weights

    def td_error_update(self, td_error):  # td_error = (q-q).detach_().abs()
        prob = td_error.squeeze().clamp(1e-6, 10).pow(self.per_alpha)
        prob = prob.cpu().numpy()
        self.update_ids(self.indices, prob)
