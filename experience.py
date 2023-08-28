import torch


class Experience:
    def __init__(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
    ):
        self.current_state = current_state
        self.action = action
        self.reward = reward
        self.next_state = next_state
