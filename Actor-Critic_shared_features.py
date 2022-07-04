import json
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format
from torch.distributions import Categorical

from environment.action_list import ACTION_LIST
from environment.constants import ALL_POWERS
from environment.observation_utils import get_observation_length, MAX_ORDERS, get_observation
from environment.order_utils import loc_to_ix, id_to_order, ORDER_SIZE, ix_to_order

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


class Brain(nn.Module):
    def __init__(self, state_size, action_size):
        super(Brain, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.actor_head = nn.Linear(256, self.action_size)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x):
        output = F.relu(self.linear1(x))
        output = F.relu(self.linear2(output))

        actor_output = self.actor_head(output)
        distribution = Categorical(F.softmax(actor_output, dim=-1))

        value = F.relu(self.critic_head(output))

        return distribution, value


class ActorCriticPlayer:
    def __init__(self, state_size, action_size, gamma=0.99):
        self.model = Brain(state_size, action_size).to(device)
        self.gamma = gamma


def train(max_steps, num_episodes, learning_rate=0.99):
    actor_state_size = get_observation_length() + 2
    critic_state_size = get_observation_length() + 1
    action_size = len(ACTION_LIST)

    player = ActorCriticPlayer(actor_state_size, critic_state_size, action_size)
    optimizer_a = optim.Adam(player.model.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        print(episode)
        game = Game()
        prev_score = {power_name: len(game.get_state()["centers"][power_name]) for power_name in ALL_POWERS}

        rewards_per_power = {power_name: [] for power_name in ALL_POWERS}
        values_per_power = {power_name: [] for power_name in ALL_POWERS}
        log_probs_per_power = {power_name: [] for power_name in ALL_POWERS}

        for step in range(max_steps):
            if game.is_game_done:
                # TODO reward winner
                break

            # global independent state
            observation = get_observation(game)

            # choose actions and save log_prob and value
            for power_idx, power_name in enumerate(ALL_POWERS):
                orders = []
                critic_state = np.concatenate([[power_idx], observation])
                critic_state = torch.FloatTensor(critic_state).to(device)
                value = player.model(critic_state)
                values_per_power[power_name].append(value)

                step_log_probs = []

                for loc in game.get_orderable_locations(power_name):
                    # add power and relevant location to observation
                    actor_state = np.concatenate([[power_idx, loc_to_ix(loc)], observation])
                    actor_state = torch.FloatTensor(actor_state).to(device)
                    dist = player.actor(actor_state)

                    action = dist.sample()

                    log_prob = dist.log_prob(action).unsqueeze(0)
                    step_log_probs.append(log_prob)

                    order = ix_to_order(action)
                    if order in game.get_all_possible_orders()[loc]:
                        orders.append(order)

                log_probs_per_power[power_name].append(step_log_probs)

                game.set_orders(power_name, orders)

            game.process()

            # save rewards
            score = {power_name: len(game.get_state()["centers"][power_name]) for power_name in ALL_POWERS}
            for power_name in ALL_POWERS:
                rewards_per_power[power_name].append(
                    torch.tensor(
                        np.subtract(score[power_name], prev_score[power_name]),
                        dtype=torch.float,
                        device=device))
            prev_score = score

        # calculate qvals and update models
        new_observation = get_observation(game)
        for power_idx, power_name in enumerate(ALL_POWERS):
            log_probs = log_probs_per_power[power_name]
            values = values_per_power[power_name]
            rewards = rewards_per_power[power_name]

            new_critic_state = np.concatenate([[power_idx], new_observation])
            new_critic_state = torch.FloatTensor(new_critic_state).to(device)
            qval = player.critic(new_critic_state)

            qvals = np.zeros(len(values))
            for t in reversed(range(len(rewards))):
                qval = rewards[t] + player.gamma * qval
                qvals[t] = qval

            # update actor critic
            values = torch.cat(values)
            qvals = torch.FloatTensor(qvals).to(device)

            advantage = qvals - values

            step_actor_loss = []
            for i, step_log_probs in enumerate(log_probs):
                step_actor_loss.append((-torch.stack(step_log_probs) * advantage[i].detach()).mean())
            actor_loss = torch.stack(step_actor_loss).mean()

            critic_loss = 0.5 * advantage.pow(2).mean()

            optimizer_a.zero_grad()
            optimizer_c.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
        optimizer_a.step()
        optimizer_c.step()

        with open(f'games/game_{episode}.json', 'w') as file:
            file.write(json.dumps(to_saved_game_format(game)))


if __name__ == "__main__":
    train(max_steps=20000, num_episodes=100)
