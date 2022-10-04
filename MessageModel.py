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
from tornado import gen

from StatTracker import StatTracker
from environment.action_list import ACTION_LIST
from environment.constants import ALL_POWERS, LOCATIONS
from environment.message_list import MESSAGE_TOKENS
from environment.observation_utils import LOC_VECTOR_LENGTH, get_board_state, get_last_phase_orders
from environment.order_utils import loc_to_ix, id_to_order, ORDER_SIZE, ix_to_order, get_valid_orders, \
    get_loc_valid_orders

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.autograd.set_detect_anomaly(True)


class Encoder(nn.Module):
    def __init__(self, state_size, embed_size, transformer_layers):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        # Linear Layer: state (81*36) > encoding size (81*embed_size)
        self.state_size = state_size
        self.linear = nn.Linear(self.state_size, embed_size)

        # Torch Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

    def forward(self, x_bo, x_po):
        x = torch.cat([x_bo, x_po], -1)
        x = self.linear(x)
        x = torch.reshape(x, (-1, 1, self.embed_size))
        x = self.transformer_encoder(x)
        return x


class ForeignRelations(nn.Module):
    def __init__(self, state_size=LOC_VECTOR_LENGTH + ORDER_SIZE, num_tokens=len(MESSAGE_TOKENS), lstm_layers=2,
                 gamma=0.99):
        super(ForeignRelations, self).__init__()
        self.num_tokens = num_tokens
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.lstm_layers, 1, self.num_tokens).to(device),
                torch.zeros(self.lstm_layers, 1, self.num_tokens).to(device))


class General(nn.Module):
    def __init__(self, state_size=LOC_VECTOR_LENGTH + ORDER_SIZE, embed_size=224, transformer_layers=10, lstm_layers=2,
                 gamma=0.99):
        super(General, self).__init__()

        self.gamma = gamma

        self.embed_size = embed_size
        self.lstm_layers = lstm_layers

        # Encoder
        self.encoder = Encoder(state_size, embed_size, transformer_layers)

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
                x_pol = F.softmax(self.linearPolicy(x_pol), dim=1)
                dist[power] = torch.reshape(x_pol, (len(locs_ix), -1))

        # value
        x_value = torch.flatten(x)
        x_value = F.relu(self.linear1(x_value))
        value = F.softmax(self.linear2(x_value), dim=0)

        return dist, value


class Player:
    def __init__(self, general_path=None, ):
        self.general = General()
        self.foreignRelations = ForeignRelations()

        if model_path:
            self.general.load_state_dict(torch.load(model_path))
            self.general.eval()

        self.general.to(device)

    @gen.coroutine
    def get_orders(self, game, power_name):
        board_state = torch.Tensor(get_board_state(game)).to(device)
        prev_orders = torch.Tensor(get_last_phase_orders(game)).to(device)
        orderable_locs = game.get_orderable_locations()
        dist, _ = self.general(board_state, prev_orders, [power_name], orderable_locs)

        power_dist = dist[power_name]
        actions = filter_orders(power_dist, power_name, game)

        return [ix_to_order(ix) for ix in actions]


def train(num_episodes, learning_rate=0.99, model_path=None):
    player = Player(model_path)

    optimizer = optim.Adam(player.general.parameters(), lr=learning_rate)

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

            board_state = torch.Tensor(get_board_state(game)).to(device)
            prev_orders = torch.Tensor(get_last_phase_orders(game)).to(device)

            orderable_locs = game.get_orderable_locations()
            dist, values = player.brain(board_state, prev_orders, ALL_POWERS, orderable_locs)

            for power, value in zip(ALL_POWERS, values):
                episode_values[power].append(value)

            for power in ALL_POWERS:
                power_orders = []

                power_dist = dist[power]

                if len(power_dist) > 0:
                    actions = filter_orders(power_dist, power, game)

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


def calculate_backdrop(player, game, episode_values, episode_log_probs, episode_rewards, optimizer):
    board_state = torch.Tensor(get_board_state(game)).to(device)
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


def filter_orders(dist, power_name, game):
    orderable_locs = game.get_orderable_locations()

    # filter invalid orders
    dist_clone = dist.clone().detach()
    for i, loc in enumerate(orderable_locs[power_name]):
        order_mask = torch.ones_like(dist_clone[i], dtype=torch.bool)
        order_mask[get_loc_valid_orders(game, loc)] = False
        dist_clone[i, :] = dist_clone[i, :].masked_fill(order_mask, value=0)

    dist_clone = Categorical(dist_clone)

    actions = dist_clone.sample()
    return actions
