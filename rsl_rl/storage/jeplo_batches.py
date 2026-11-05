"""Reiterate original CTS sample order without materializing every epoch.

Upstream flattens storage in env-major order and stores all minibatch tensor
copies for PPO and student distillation. Indexing [time, env] gives exactly the
same values with O(rollout indices + one batch) additional memory.
"""
import torch

from rsl_rl.third_party.jeplo.rollout_storage_teacher_cts import RolloutStorageTeacherCTS


class StreamingCTSStorage(RolloutStorageTeacherCTS):
    def reusable_batches(self, num_mini_batches, num_epochs):
        return CTSBatchPlan(self, num_mini_batches, num_epochs)


class CTSBatchPlan:
    def __init__(self, storage, num_mini_batches, num_epochs):
        self.storage = storage
        self.num_mini_batches = num_mini_batches
        self.num_epochs = num_epochs
        teacher_count = storage.teacher_num_envs * storage.num_transitions_per_env
        student_count = storage.student_num_envs * storage.num_transitions_per_env
        self.teacher_size = teacher_count // num_mini_batches
        self.student_size = student_count // num_mini_batches if student_count else 0
        # Same two RNG calls, order and permutation reuse as upstream.
        self.teacher_indices = torch.randperm(teacher_count, requires_grad=False, device=storage.device)
        self.student_indices = (teacher_count + torch.randperm(student_count, requires_grad=False, device=storage.device)
                                if student_count else torch.empty(0, dtype=torch.long, device=storage.device))

    def __iter__(self):
        storage = self.storage
        fields = (storage.observations, storage.high_features, storage.actions,
                  storage.values, storage.advantages, storage.returns,
                  storage.actions_log_prob, storage.mu, storage.sigma)
        for _ in range(self.num_epochs):
            for i in range(self.num_mini_batches):
                indices = self.teacher_indices[i*self.teacher_size:(i+1)*self.teacher_size]
                if self.student_size:
                    indices = torch.cat((indices, self.student_indices[i*self.student_size:(i+1)*self.student_size]))
                times = indices % storage.num_transitions_per_env
                envs = indices // storage.num_transitions_per_env
                yield tuple(field[times, envs].detach() for field in fields)
