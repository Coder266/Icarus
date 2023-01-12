import time

import torch
from torch import nn as nn
from torch.nn import functional as F
from tornado import gen

from environment.action_list import ACTION_LIST
from environment.constants import LOCATIONS, ALL_POWERS
from environment.message_list import MESSAGE_LIST, ANSWER_LIST
from environment.message_utils import ix_to_msg
from environment.observation_utils import LOC_VECTOR_LENGTH, get_board_state, get_last_phase_orders
from environment.order_utils import ORDER_SIZE, loc_to_ix, ix_to_order, select_orders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

torch.autograd.set_detect_anomaly(True)


class MessagePlayer:
    def __init__(self, model_path=None, embed_size=224, msg_embed_size=100, transformer_layers=10, transformer_heads=8, lstm_size=200,
                 lstm_layers=2, press_time=3, msg_log_size=20, gamma=0.99):
        self.press_time = press_time
        self.msg_log = torch.zeros([msg_log_size, msg_embed_size])

        self.brain = Brain(embed_size=embed_size, msg_embed_size=msg_embed_size, transformer_layers=transformer_layers,
                           transformer_heads=transformer_heads, lstm_size=lstm_size, lstm_layers=lstm_layers,
                           msg_log_size=msg_log_size, gamma=gamma)

        if model_path:
            self.brain.load_state_dict(torch.load(model_path))
            self.brain.eval()

        self.brain.to(device)

    @gen.coroutine
    def get_orders(self, game, power_name):
        start_time = time.time()

        board_state = torch.Tensor(get_board_state(game.get_state())).to(device)
        prev_orders = torch.Tensor(get_last_phase_orders(game)).to(device)

        self.send_press(game, power_name, board_state, prev_orders)

        while not time.time() - start_time >= self.press_time:
            self.check_messages(game, power_name, board_state, prev_orders)

        orderable_locs = game.get_orderable_locations()

        dist, _ = self.brain(board_state, prev_orders, self.msg_log, [power_name], orderable_locs)

        actions = select_orders(dist[power_name], power_name, game, orderable_locs)

        return [ix_to_order(ix) for ix in actions]

    def send_press(self, game, power_name, board_state, prev_orders):
        msgs = self.brain.forward_msgs(board_state, prev_orders, self.msg_log, game, power_name)

        for msg in msgs:
            self.send_message(msg)

    def check_messages(self, game, power_name, board_state, prev_orders):
        # todo logic to check for incoming messages
        incoming_messages = []

        for msg in incoming_messages:
            self.add_message(msg)
            self.reply_press(game, power_name, board_state, prev_orders, msg)

    def reply_press(self, game, power_name, board_state, prev_orders, last_message):
        msg = self.brain.forward_answer(board_state, prev_orders, self.msg_log, game, power_name, last_message)
        self.send_message(msg)

    def add_message(self, msg):
        self.msg_log = torch.cat((self.msg_log[1:], torch.Tensor([msg])))

    def send_message(self, msg):
        self.add_message(msg)
        # todo logic to send messages, check target etc
        pass


class Brain(nn.Module):
    def __init__(self, state_size=LOC_VECTOR_LENGTH + ORDER_SIZE, embed_size=224, msg_embed_size=100, transformer_layers=10,
                 transformer_heads=8, lstm_size=200, lstm_layers=2, msg_log_size=20, gamma=0.99):
        super(Brain, self).__init__()

        self.gamma = gamma
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers

        self.msg_embedding = nn.Embedding(len(MESSAGE_LIST), msg_embed_size)

        # Encoder
        self.encoder = Encoder(state_size, embed_size, transformer_layers, transformer_heads)

        # Policy Network
        # LSTM Decoder: encoded state (embed_size) > action probabilities (len(ACTION_LIST))
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(embed_size * 2, lstm_size, num_layers=lstm_layers)
        self.linearPolicy = nn.Linear(lstm_size, len(ACTION_LIST))

        # Value Network
        self.linear1 = nn.Linear(len(LOCATIONS) * embed_size + embed_size, embed_size)
        self.linear2 = nn.Linear(embed_size, len(ALL_POWERS))

        # messages
        self.msgEmbedLinear = nn.Linear(msg_log_size * msg_embed_size, embed_size)
        self.msgOutputLinear = nn.Linear(len(LOCATIONS) * embed_size + embed_size, len(MESSAGE_LIST))
        self.msgReplyLinear = nn.Linear(embed_size, len(ANSWER_LIST))

    def init_hidden(self):
        return (torch.zeros(self.lstm_layers, 1, self.lstm_size).to(device),
                torch.zeros(self.lstm_layers, 1, self.lstm_size).to(device))

    def forward(self, x_bo, x_po, msg_log, powers, locs_by_power):
        x = self.encoder(x_bo, x_po)

        # TODO fix shape issues probably
        msg_state_embed = F.relu(self.msgEmbedLinear(torch.flatten(msg_log)))

        # policy
        dist = {}
        for power in powers:
            if not locs_by_power[power]:
                dist[power] = torch.Tensor([]).to(device)
            else:
                self.hidden = self.init_hidden()
                locs_ix = [loc_to_ix(loc) for loc in locs_by_power[power]]
                locs_emb = x[locs_ix]
                x_pol, self.hidden = self.lstm(torch.cat([locs_emb, msg_state_embed]), self.hidden)
                x_pol = self.linearPolicy(x_pol)
                dist[power] = torch.reshape(x_pol, (len(locs_ix), -1))

        # value
        x_value = torch.cat([torch.flatten(x), msg_state_embed])
        x_value = F.relu(self.linear1(x_value))
        value = self.linear2(x_value)

        return dist, value

    def forward_msgs(self, x_bo, x_po, msg_log, game, power_name):
        # calculates the probability of sending each message and sends all above 0.5
        x = self.encoder(x_bo, x_po)
        msg_state_embed = F.relu(self.msgEmbedLinear(torch.flatten(msg_log)))
        x = torch.cat([torch.flatten(x), msg_state_embed])
        x = F.sigmoid(self.msgOutputLinear(x))
        x = x.ge(0.5).nonzero()

        return [ix_to_msg(msg, game, power_name) for msg in x]

    def forward_answer(self, x_bo, x_po, msg_log, game, power_name, last_message):
        # calculates the best response to a message, yes, rej or no answer
        x = self.encoder(x_bo, x_po)
        msg_state_embed = F.relu(self.msgEmbedLinear(torch.flatten(msg_log)))
        x = torch.cat([torch.flatten(x), msg_state_embed])
        x = F.softmax(self.msgReplyLinear(x))

        return ix_to_msg(torch.multinomial(x, 1), game, power_name, last_message)


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
