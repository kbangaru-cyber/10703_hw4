import gymnasium as gym
import numpy as np
import torch
from torch import nn
print('device:', 'cuda' if torch.cuda.is_available() else 'cpu')

import argparse
import imageio
from modules import PolicyNet
from simple_network import SimpleNet
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os
try:
    import wandb
except ImportError:
    wandb = None

import warnings
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1" 

def pick_device() -> str:
    """
    Use CUDA if available and kernels actually run; otherwise fall back to CPU.
    This avoids crashes on very new GPUs whose SM arch isn't baked into your wheel.
    """
    if torch.cuda.is_available():
        try:
            torch.empty(1, device="cuda").fill_(0) 
            return "cuda"
        except Exception as e:
            print("CUDA present but kernels unsupported -> using CPU. Detail:", e)
    return "cpu"

class TrainDaggerBC:

    def __init__(self, env, model, optimizer, states, actions, expert_model=None, device="cpu", mode="DAgger"):
        """
        Initializes the TrainDAgger class. Creates necessary data structures.

        Args:
            env: an OpenAI Gym environment.
            model: the model to be trained.
            expert_model: the expert model that provides the expert actions.
            device: the device to be used for training.
            mode: the mode to be used for training. Either "DAgger" or "BC".

        """
        self.env = env
        self.model = model
        self.expert_model = expert_model
        self.optimizer = optimizer
        self.device = device
        self.model.set_device(self.device)
        self.mode = mode

        if self.mode == "BC":
            self.states, self.actions, self.timesteps = [], [], []
            for traj_idx in range(states.shape[0]):
                mask = (states[traj_idx].sum(axis=1) != 0) 
                s_traj = states[traj_idx][mask]
                a_traj = actions[traj_idx][mask]
                self.states.append(s_traj)
                self.actions.append(a_traj)
                self.timesteps.append(np.arange(len(s_traj), dtype=np.int64))

            self.states = np.concatenate(self.states, axis=0)
            self.actions = np.concatenate(self.actions, axis=0)
            self.timesteps = np.concatenate(self.timesteps, axis=0)

            self.actions = np.clip(self.actions, -1.0, 1.0)

        else:
            self.expert_model = self.expert_model.to(self.device)
            self.expert_model.eval()
            self.states = None
            self.actions = None
            self.timesteps = None

    def generate_trajectory(self, env, policy, render=False):
        """Collects one rollout from the policy in an environment. The environment
        should implement the OpenAI Gym interface. A rollout ends when done=True. The
        number of states and actions should be the same, so you should not include
        the final state when done=True.

        Args:
            env: an OpenAI Gym environment.
            policy: The output of a deep neural network
            render: Whether to store frames from the environment
            Returns:
            states: a list of states visited by the agent.
            actions: a list of actions taken by the agent. Note that these actions should never actually be trained on...
            timesteps: a list of integers, where timesteps[i] is the timestep at which states[i] was visited.
            rewards: list of rewards given by the environment
            rgbs: list of rgb images from the environment for each timestep
        """

        states, old_actions, timesteps, rewards, rgbs = [], [], [], [], []

        done, trunc = False, False
        cur_state, _ = env.reset()
        if render:
            rgbs.append(env.render())

        t = 0
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            while (not done) and (not trunc):
                p = policy(
                    torch.from_numpy(cur_state).to(self.device).float().unsqueeze(0),
                    torch.tensor(t, device=self.device, dtype=torch.long).unsqueeze(0),
                )
                a = p.squeeze(0).detach().cpu().numpy()
                a = np.clip(a, -1.0, 1.0) 

                next_state, reward, done, trunc, _ = env.step(a)

                states.append(cur_state)
                old_actions.append(a)
                timesteps.append(t)
                rewards.append(reward)

                if render:
                    rgbs.append(env.render())

                t += 1
                cur_state = next_state

        if was_training:
            self.model.train()

        return states, old_actions, timesteps, rewards, rgbs

    def call_expert_policy(self, state):
        """
        Calls the expert policy to get an action.

        Args:
            state: the current state of the environment.
        """
        # takes in a np array state and returns an np array action
        with torch.no_grad():
            s = torch.tensor(state[None, :], dtype=torch.float32, device=self.device)
            action = self.expert_model.choose_action(s, deterministic=True).detach().cpu().numpy()[0]
            action = np.clip(action, -1.0, 1.0)
        return action

    def update_training_data(self, num_trajectories_per_batch_collection=20):
        """
        Updates the training data by collecting trajectories from the current policy and the expert policy.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.

        NOTE: you will need to call self.generate_trajectory and self.call_expert_policy in this function.
        NOTE: you should update self.states, self.actions, and self.timesteps in this function.
        """
        # BEGIN STUDENT SOLUTION
        assert self.mode == "DAgger", "update_training_data is only for DAgger"

        rewards_per_traj = []
        collected_states, collected_actions, collected_timesteps = [], [], []

        for _ in range(num_trajectories_per_batch_collection):
            states, _, ts, rewards, _ = self.generate_trajectory(self.env, self.model, render=False)
            rewards_per_traj.append(float(np.sum(rewards)))

            # label each visited state with expert
            expert_actions = [self.call_expert_policy(s) for s in states]

            collected_states.append(np.asarray(states, dtype=np.float32))
            collected_actions.append(np.asarray(expert_actions, dtype=np.float32))
            collected_timesteps.append(np.arange(len(states), dtype=np.int64))

        if len(collected_states) > 0:
            S = np.concatenate(collected_states, axis=0)
            A = np.concatenate(collected_actions, axis=0)
            T = np.concatenate(collected_timesteps, axis=0)
        else:
            S = np.zeros((0, 24), dtype=np.float32)
            A = np.zeros((0, 4), dtype=np.float32)
            T = np.zeros((0,), dtype=np.int64)

        A = np.clip(A, -1.0, 1.0)

        if self.states is None:
            self.states, self.actions, self.timesteps = S, A, T
        else:
            self.states = np.concatenate([self.states, S], axis=0)
            self.actions = np.concatenate([self.actions, A], axis=0)
            self.timesteps = np.concatenate([self.timesteps, T], axis=0)

        # END STUDENT SOLUTION

        return rewards_per_traj

    def generate_trajectories(self, num_trajectories_per_batch_collection=20):
        """
        Runs inference for a certain number of trajectories. Use for behavior cloning.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.
        
        NOTE: you will need to call self.generate_trajectory in this function.
        """
        rewards = []
        # BEGIN STUDENT SOLUTION
        for _ in range(num_trajectories_per_batch_collection):
            _, _, _, r, _ = self.generate_trajectory(self.env, self.model, render=False)
            rewards.append(float(np.sum(r)))

        # END STUDENT SOLUTION

        return rewards

    def train(
        self, 
        num_batch_collection_steps=20, 
        num_training_steps_per_batch_collection=1000, 
        num_trajectories_per_batch_collection=20, 
        batch_size=64, 
        print_every=500, 
        save_every=10000, 
        wandb_logging=False
    ):
        """
        Train the model using BC or DAgger

        Args:
            num_batch_collection_steps: the number of times to collecta batch of trajectories from the current policy.
            num_training_steps_per_batch_collection: the number of times to train the model per batch collection.
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy per batch.
            batch_size: the batch size to use for training.
            print_every: how often to print the loss during training.
            save_every: how often to save the model during training.
            wandb_logging: whether to log the training to wandb.

        NOTE: for BC, you will need to call the self.training_step function and self.generate_trajectories function.
        NOTE: for DAgger, you will need to call the self.training_step and self.update_training_data function.
        """

        # BEGIN STUDENT SOLUTION
        total_updates = num_batch_collection_steps * num_training_steps_per_batch_collection
        losses = np.zeros(total_updates, dtype=np.float32)
        mean_rewards, median_rewards, max_rewards = [], [], []

        self.model.train()
        cur = 0
        for b in range(num_batch_collection_steps):
            if self.mode == "BC":
                for s in range(num_training_steps_per_batch_collection):
                    loss = self.training_step(batch_size=batch_size)
                    losses[cur] = loss
                    cur += 1
                    if (s + 1) % print_every == 0:
                        print(f"[BC   batch {b+1:02d}] step {s+1:04d} | loss {loss:.4f}")
                batch_returns = self.generate_trajectories(num_trajectories_per_batch_collection)
            else:
                batch_returns = self.update_training_data(num_trajectories_per_batch_collection)
                for s in range(num_training_steps_per_batch_collection):
                    loss = self.training_step(batch_size=batch_size)
                    losses[cur] = loss
                    cur += 1
                    if (s + 1) % print_every == 0:
                        print(f"[DAggr batch {b+1:02d}] step {s+1:04d} | loss {loss:.4f}")

            m, md, mx = float(np.mean(batch_returns)), float(np.median(batch_returns)), float(np.max(batch_returns))
            mean_rewards.append(m); median_rewards.append(md); max_rewards.append(mx)
            print(f"[{self.mode} batch {b+1:02d}] mean {m:.1f} | median {md:.1f} | max {mx:.1f}")

        # END STUDENT SOLUTION
        x_axis = np.arange(0, len(mean_rewards)) * num_training_steps_per_batch_collection
        plt.figure()
        plt.plot(x_axis, mean_rewards, label="mean rewards")
        plt.plot(x_axis, median_rewards, label="median rewards")
        plt.plot(x_axis, max_rewards, label="max rewards")
        plt.legend()
        plt.savefig(f"{self.mode}_rewards.png")

        plt.figure()
        plt.plot(np.arange(0, len(losses)), losses, label="training loss")
        plt.legend()
        plt.savefig(f"{self.mode}_losses.png")

        return losses

    def training_step(self, batch_size):
        """
        Simple training step implementation

        Args:
            batch_size: the batch size to use for training.
        """
        states, actions, timesteps = self.get_training_batch(batch_size=batch_size)
        loss_fn = nn.MSELoss()
        self.optimizer.zero_grad(set_to_none=True)
        pred = self.model(states, timesteps)
        loss = loss_fn(pred, actions)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def get_training_batch(self, batch_size=64):
        """
        get a training batch

        Args:
            batch_size: the batch size to use for training.
        """
        # get random states, actions, and timesteps
        assert self.states is not None and len(self.states) > 0, "No training data available."
        idx = np.random.choice(len(self.states), size=batch_size, replace=False)
        states = torch.tensor(self.states[idx], device=self.device).float()
        actions = torch.tensor(self.actions[idx], device=self.device).float()
        timesteps = torch.tensor(self.timesteps[idx], device=self.device).long()
        return states, actions, timesteps

def run_training(dagger: bool):
    """
    Simple Run Training Function
    """

    env = gym.make('BipedalWalker-v3', render_mode='rgb_array') # , render_mode="rgb_array"
    with open(f"data/states_BC.pkl", "rb") as f:
        states = pickle.load(f)
    with open(f"data/actions_BC.pkl", "rb") as f:
        actions = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dagger:
        # Load expert model
        expert_model = PolicyNet(24, 4).to(device)
        ckpt_path = os.path.join("data", "models", "super_expert_PPO_model.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        expert_model.load_state_dict(ckpt["PolicyNet"])
        expert_model.eval()
        # BEGIN STUDENT SOLUTION
        model = SimpleNet(
            state_dim=24, action_dim=4,
            hidden_layer_dimension=128, max_episode_length=1600,
            device=device
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

        trainer = TrainDaggerBC(
            env=env, model=model, optimizer=optimizer,
            states=states, actions=actions,   # not used in DAgger loop
            expert_model=expert_model, device=device, mode="DAgger",
        )

        losses = trainer.train(
            num_batch_collection_steps=20,
            num_training_steps_per_batch_collection=1000,
            num_trajectories_per_batch_collection=20,
            batch_size=128,
        )

        print(f"Final {trainer.mode} loss:", losses[-1])
        # END STUDENT SOLUTION
        _, _, _, rewards, rgbs = trainer.generate_trajectory(trainer.env, trainer.model, render=True)
        print(f"DAgger rollout return: {sum(rewards):.1f}")
        imageio.mimsave('gifs_DAgger.gif', rgbs, fps=33)

    else:
        # BEGIN STUDENT SOLUTION
        model = SimpleNet(
            state_dim=24, action_dim=4,
            hidden_layer_dimension=128, max_episode_length=1600,
            device=device
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

        trainer = TrainDaggerBC(
            env=env, model=model, optimizer=optimizer,
            states=states, actions=actions,
            expert_model=None, device=device, mode="BC",
        )

        losses = trainer.train(
            num_batch_collection_steps=20,
            num_training_steps_per_batch_collection=1000,
            num_trajectories_per_batch_collection=20,
            batch_size=128,
        )

        print(f"Final {trainer.mode} loss:", losses[-1])
        # END STUDENT SOLUTION
        _, _, _, rewards, rgbs = trainer.generate_trajectory(trainer.env, trainer.model, render=True)
        print(f"BC rollout return: {sum(rewards):.1f}")
        imageio.mimsave('gifs_BC.gif', rgbs, fps=33)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dagger', action='store_true', help="Run DAgger instead of BC")
    args = parser.parse_args()
    run_training(args.dagger)
