import json

import numpy as np
import torch
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format
from torch import optim as optim
from torch.distributions import Categorical
from torch.nn import functional as F

from Player import Player, device
from StatTracker import StatTracker
from environment.constants import ALL_POWERS
from environment.observation_utils import get_board_state, get_last_phase_orders
from environment.order_utils import ix_to_order, select_orders


def train_rl(num_episodes, learning_rate=0.001, model_path=None, gamma=0.99):
    def calculate_backdrop(player, game, episode_values, episode_log_probs, episode_rewards, optimizer):
        board_state = torch.Tensor(get_board_state(game.get_state())).to(device)
        prev_orders = torch.Tensor(get_last_phase_orders(game)).to(device)

        orderable_locs = game.get_orderable_locations()
        _, new_values = player(board_state, prev_orders, ALL_POWERS, orderable_locs)

        for power_idx, power in enumerate(ALL_POWERS):
            log_probs = episode_log_probs[power]
            values = episode_values[power]
            rewards = episode_rewards[power]
            qval = new_values[power_idx]

            qvals = np.zeros(len(values))
            for t in reversed(range(len(rewards))):
                qval = rewards[t] + gamma * qval
                qvals[t] = qval

            # update actor critic
            values = torch.stack(values)
            qvals = torch.FloatTensor(qvals).to(device)

            advantage = qvals - values

            step_actor_loss = []
            for step, step_log_probs in enumerate(log_probs):
                step_actor_loss.append((-step_log_probs * advantage[step].detach()).mean())
            actor_loss = torch.stack(step_actor_loss).mean()

            critic_loss = 0.5 * advantage.pow(2).mean()

            optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            critic_loss.backward(retain_graph=True)

    player = Player(model_path)

    player.brain.train()

    optimizer = optim.Adam(player.brain.parameters(), lr=learning_rate)

    stat_tracker = StatTracker()

    for episode in range(num_episodes):
        game = Game()
        stat_tracker.new_game(game)

        prev_score = {power_name: len(game.get_state()["centers"][power_name]) for power_name in ALL_POWERS}
        episode_values = {power_name: [] for power_name in ALL_POWERS}
        episode_log_probs = {power_name: [] for power_name in ALL_POWERS}
        episode_rewards = {power_name: [] for power_name in ALL_POWERS}

        step = 0
        while not game.is_game_done:
            step += 1

            board_state = torch.Tensor(get_board_state(game.get_state())).to(device)
            prev_orders = torch.Tensor(get_last_phase_orders(game)).to(device)

            orderable_locs = game.get_orderable_locations()
            dist, values = player.brain(board_state, prev_orders, ALL_POWERS, orderable_locs)

            for power, value in zip(ALL_POWERS, values):
                episode_values[power].append(value)

            for power in ALL_POWERS:
                power_orders = []

                power_dist = F.softmax(dist[power], dim=1)

                if len(power_dist) > 0:
                    actions = select_orders(dist[power], game, power, orderable_locs)

                    power_dist = Categorical(power_dist)
                    episode_log_probs[power].append(power_dist.log_prob(actions))

                    power_orders = [ix_to_order(ix) for ix in actions]

                game.set_orders(power, power_orders)

            game.process()

            score = {power_name: len(game.get_state()["centers"][power_name]) for power_name in ALL_POWERS}

            for power_name in ALL_POWERS:
                power_reward = np.subtract(score[power_name], prev_score[power_name])
                if game.is_game_done and power_name in game.outcome:
                    if score[power_name] >= 18:
                        power_reward = 34
                    else:
                        power_reward = score[power_name] * 34 / sum(score.values())

                episode_rewards[power_name].append(
                    torch.tensor(
                        power_reward,
                        dtype=torch.float,
                        device=device))
            prev_score = score

            stat_tracker.update(game)

        calculate_backdrop(player.brain, game, episode_values, episode_log_probs, episode_rewards, optimizer)

        print(f'Game Done\nEpisode {episode}, Step {step}\nScore: {score}\nWinners:{game.outcome[1:]}')
        stat_tracker.end_game()

        if episode % 10 == 0:
            stat_tracker.plot_game()
            stat_tracker.plot_wins()

            with open(f'games/game_{episode}.json', 'w') as file:
                file.write(json.dumps(to_saved_game_format(game)))

            torch.save(player.brain.state_dict(), f'models/model_{episode}.pth')
