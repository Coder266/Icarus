import torch
from torch import nn as nn
from torch.nn import functional as F
from tornado import gen

from environment.action_list import ACTION_LIST
from environment.constants import LOCATIONS, ALL_POWERS
from environment.observation_utils import LOC_VECTOR_LENGTH, get_board_state, get_last_phase_orders
from environment.order_utils import ORDER_SIZE, loc_to_ix, ix_to_order, filter_orders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

torch.autograd.set_detect_anomaly(True)


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
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(embed_size, embed_size, lstm_layers)
        self.linearPolicy = nn.Linear(embed_size, len(ACTION_LIST))

        # Value Network
        self.linear1 = nn.Linear(len(LOCATIONS) * embed_size, embed_size)
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
                # TODO insert loc_emb, masked previous loc_emb
                x_pol, self.hidden = self.lstm(locs_emb, self.hidden)
                x_pol = self.linearPolicy(x_pol)
                dist[power] = torch.reshape(x_pol, (len(locs_ix), -1))

        # value
        x_value = torch.flatten(x)
        x_value = F.relu(self.linear1(x_value))
        value = self.linear2(x_value)

        return dist, value


class Encoder(nn.Module):
    def __init__(self, state_size, embed_size, transformer_layers, transformer_heads):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.state_size = state_size

        self.linear = nn.Linear(self.state_size, embed_size)

        self.positional_bias = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(len(LOCATIONS), 1, embed_size)))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=transformer_heads, dim_feedforward=embed_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

    def forward(self, x_bo, x_po):
        x = torch.cat([x_bo, x_po], -1)
        x = self.linear(x)
        x = torch.reshape(x, (-1, 1, self.embed_size))
        x = x + self.positional_bias
        x = self.transformer_encoder(x)
        return x