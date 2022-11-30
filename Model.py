import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from diplomacy import Game
from diplomacy.utils.export import to_saved_game_format
from torch.distributions import Categorical
from tornado import gen
import jsonlines

from StatTracker import StatTracker
from environment.action_list import ACTION_LIST
from environment.constants import ALL_POWERS, LOCATIONS, POWER_ACRONYMS_LIST
from environment.observation_utils import LOC_VECTOR_LENGTH, get_board_state, get_last_phase_orders, phase_orders_to_rep
from environment.order_utils import ORDER_SIZE, ix_to_order, get_loc_valid_orders, order_to_ix, loc_to_ix

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.autograd.set_detect_anomaly(True)


class Encoder(nn.Module):
    def __init__(self, state_size, embed_size, transformer_layers, transformer_heads):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        # Linear Layer: state (81*36) > encoding size (81*embed_size)
        self.state_size = state_size
        self.linear = nn.Linear(self.state_size, embed_size)

        # Torch Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=transformer_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

    def forward(self, x_bo, x_po):
        x = torch.cat([x_bo, x_po], -1)
        x = self.linear(x)
        x = torch.reshape(x, (-1, 1, self.embed_size))
        x = self.transformer_encoder(x)
        return x


class Brain(nn.Module):
    def __init__(self, state_size=LOC_VECTOR_LENGTH + ORDER_SIZE, embed_size=224, transformer_layers=10,
                 transformer_heads=8, lstm_layers=2, gamma=0.99):
        super(Brain, self).__init__()

        self.gamma = gamma

        self.embed_size = embed_size
        self.lstm_layers = lstm_layers

        # Encoder
        self.encoder = Encoder(state_size, embed_size, transformer_layers, transformer_heads)

        # Policy Network
        # LSTM Decoder: encoded state (embed_size) > action probabilities (len(ACTION_LIST))
        self.lstm = nn.LSTM(embed_size, embed_size, lstm_layers)
        self.hidden = self.init_hidden()

        self.linearPolicy = nn.Linear(embed_size, len(ACTION_LIST))

        # Value Network
        # Linear, Relu: (81 * embed_size) > embed_size
        self.linear1 = nn.Linear(len(LOCATIONS) * embed_size, embed_size)

        # Linear, Softmax: embed_size > # of players
        self.linear2 = nn.Linear(embed_size, len(ALL_POWERS))

    def init_hidden(self):
        return (torch.zeros(self.lstm_layers, 1, self.embed_size).to(device),
                torch.zeros(self.lstm_layers, 1, self.embed_size).to(device))

    def forward(self, x_bo, x_po, powers, locs_by_power):
        x = self.encoder(x_bo, x_po)

        # policy
        dist = {}
        for power in powers:
            if not locs_by_power[power]:
                dist[power] = torch.Tensor([]).to(device)
            else:
                self.hidden = self.init_hidden()
                locs_ix = [loc_to_ix(loc) for loc in locs_by_power[power]]
                locs_emb = x[locs_ix]
                x_pol, self.hidden = self.lstm(locs_emb, self.hidden)
                x_pol = self.linearPolicy(x_pol)
                dist[power] = torch.reshape(x_pol, (len(locs_ix), -1))

        # value
        x_value = torch.flatten(x)
        x_value = F.relu(self.linear1(x_value))
        value = self.linear2(x_value)

        return dist, value


class Player:
    def __init__(self, model_path=None, embed_size=224, transformer_layers=10, transformer_heads=8, lstm_layers=2,
                 gamma=0.99):
        self.brain = Brain(embed_size=embed_size, transformer_layers=transformer_layers,
                           transformer_heads=transformer_heads, lstm_layers=lstm_layers, gamma=gamma)

        if model_path:
            self.brain.load_state_dict(torch.load(model_path))
            self.brain.eval()

        self.brain.to(device)

    @gen.coroutine
    def get_orders(self, game, power_name):
        board_state = torch.Tensor(get_board_state(game.get_state())).to(device)
        prev_orders = torch.Tensor(get_last_phase_orders(game)).to(device)
        orderable_locs = game.get_orderable_locations()
        dist, _ = self.brain(board_state, prev_orders, [power_name], orderable_locs)

        actions = filter_orders(dist[power_name], power_name, game)

        return [ix_to_order(ix) for ix in actions]


def train_rl(num_episodes, learning_rate=0.001, model_path=None):
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
                qval = rewards[t] + player.gamma * qval
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
                    actions = filter_orders(dist[power], power, game)

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


def train_sl(file_paths, dist_learning_rate=0.001, value_learning_rate=1e-6, model_path=None):
    def sort_orders(orderable_locs, orders):
        sorted_orders_by_power = {}
        for power in orderable_locs.keys():
            sorted_orders = []
            for loc in orderable_locs[power]:
                done = False
                if orders[power] is None:
                    sorted_orders.append('WAIVE')
                else:
                    for order in orders[power]:
                        if order != 'WAIVE' and loc in order.split()[1]:
                            sorted_orders.append(order)
                            done = True
                            break
                    if not done:
                        sorted_orders.append('WAIVE')
            sorted_orders_by_power[power] = sorted_orders
        return sorted_orders_by_power

    player = Player(model_path)

    player.brain.train()

    optimizer = optim.Adam(
        [
            {"params": player.brain.encoder.parameters()},
            {"params": player.brain.lstm.parameters()},
            {"params": player.brain.linearPolicy.parameters()},
            {"params": player.brain.linear1.parameters(), "lr": value_learning_rate},
            {"params": player.brain.linear2.parameters(), "lr": value_learning_rate}
        ],
        lr=dist_learning_rate
    )
    criterion = nn.CrossEntropyLoss()

    running_dist_loss = 0.0
    running_value_loss = 0.0
    for epoch in range(10):
        input_count = 0

        for path in file_paths:
            with jsonlines.open(path) as reader:
                for game_idx, obj in enumerate(reader):
                    note = obj['phases'][-1]['state']['note'].split(': ')

                    if note[0] == 'Victory by':
                        final_score = [int(note[1] == power) * 34 for power in POWER_ACRONYMS_LIST]
                    else:
                        last_phase = obj['phases'][-1]
                        # final_score should be
                        # final_score = [score/sum(final_score) * 34 for score in final_score]
                        # but it is not needed here since we only need the proportion
                        final_score = [len(last_phase['state']['centers'][power])
                                       if power in last_phase['state']['centers']
                                       else 0 for power in ALL_POWERS]

                    last_phase_orders = []

                    for phase in obj['phases']:
                        input_count += 1

                        optimizer.zero_grad()

                        board_state = get_board_state(phase['state'])
                        prev_orders = phase_orders_to_rep(last_phase_orders)
                        powers = [k for k, i in phase['orders'].items() if i]
                        orderable_locs = {power: [unit.split()[1] for unit in units]
                                          for power, units in phase['state']['units'].items()}
                        orders = sort_orders(orderable_locs, phase['orders'])

                        last_phase_orders = orders

                        dist, value_outputs = player.brain(torch.Tensor(board_state).to(device),
                                                           torch.Tensor(prev_orders).to(device),
                                                           powers,
                                                           orderable_locs)

                        dist_outputs = [probs for power in powers for probs in dist[power]]
                        dist_labels = [order_to_ix(order) for power in powers for order in orders[power]]

                        # remove unsupported orders (ex. convoys longer than 4)
                        to_remove = [i for i, label in enumerate(dist_labels) if label is None or label == 0]
                        for index in sorted(to_remove, reverse=True):
                            del dist_outputs[index]
                            del dist_labels[index]

                        if sum(final_score) != 0:
                            value_labels = [score/sum(final_score) for score in final_score]
                        else:
                            value_labels = [0] * 7

                        if dist_labels:
                            dist_loss = criterion(torch.stack(dist_outputs).to(device),
                                                  torch.LongTensor(dist_labels).to(device))
                            dist_loss.backward(retain_graph=True)

                        value_loss = criterion(value_outputs.reshape(1, -1),
                                               torch.Tensor(value_labels).to(device).reshape(1, -1))
                        value_loss.backward()
                        optimizer.step()

                        running_dist_loss += dist_loss.item()
                        running_value_loss += value_loss.item()
                    
                    print(f'[{epoch + 1}, {game_idx + 1}] dist loss: {running_dist_loss / input_count:.3f},'
                         f' value loss: {running_value_loss / input_count:.3f}')
                    running_dist_loss = 0.0
                    running_value_loss = 0.0
                    input_count = 0
                    
                    if game_idx % 100 == 99:
                        torch.save(player.brain.state_dict(), f'models/sl_model_DipNet_{epoch + 1}_{game_idx}.pth')


def filter_orders(dist, power_name, game):
    orderable_locs = game.get_orderable_locations()

    dist_clone = dist.clone().detach()
    order_mask = torch.ones_like(dist_clone, dtype=torch.bool)
    for i, loc in enumerate(orderable_locs[power_name]):
        order_mask[i, get_loc_valid_orders(game, loc)] = False
    # dist_clone[i, :] = dist_clone[i, :].masked_fill(order_mask, value=0)

    state = game.get_state()

    n_builds = abs(state['builds'][power_name]['count'])

    if n_builds > 0:
        dist_clone = F.softmax(dist_clone.reshape(1, -1))

        dist_clone[:, 0] = 0

        dist_clone = dist_clone.masked_fill(order_mask.reshape(1, -1), value=0)

        actions = [ix % len(dist[0]) for ix in torch.multinomial(dist_clone, n_builds)[0]]
    else:
        dist_clone = F.softmax(dist_clone, dim=1)

        dist_clone = dist_clone.masked_fill(order_mask, value=0)

        dist_clone = Categorical(dist_clone)

        actions = dist_clone.sample()

    return actions
