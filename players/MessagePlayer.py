import asyncio

import time

import torch
from torch import nn as nn
from torch.nn import functional as F

from environment.action_list import ACTION_LIST
from environment.constants import LOCATIONS, ALL_POWERS
from environment.message_list import MESSAGE_LIST, ANSWER_LIST
from environment.message_utils import filter_messages, get_daide_msg_ix, get_msg_ixs_from_daide_reply, \
    is_daide_msg_reply, \
    send_message, split_DMZ
from environment.observation_utils import LOC_VECTOR_LENGTH, get_board_state, get_last_phase_orders
from environment.order_utils import ORDER_SIZE, loc_to_ix, ix_to_order, select_orders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

torch.autograd.set_detect_anomaly(True)


class MessagePlayer:
    def __init__(self, model_path=None, gunboat_model_path=None, embed_size=224, msg_embed_size=100,
                 transformer_layers=5, transformer_heads=8, lstm_size=200, lstm_layers=2, press_time=30,
                 msg_log_size=20):
        self.press_time = press_time
        self.msg_logs = {power: torch.zeros([msg_log_size, msg_embed_size]).to(device) for power in ALL_POWERS}

        self.brain = Brain(embed_size=embed_size, msg_embed_size=msg_embed_size, transformer_layers=transformer_layers,
                           transformer_heads=transformer_heads, lstm_size=lstm_size, lstm_layers=lstm_layers,
                           msg_log_size=msg_log_size)

        if model_path:
            self.brain.load_state_dict(torch.load(model_path))
            self.brain.eval()
        elif gunboat_model_path:
            gunboat_model = torch.load(gunboat_model_path)

            layers = ['linear1', 'linear2', 'encoder']
            layer_keys = [key for key in gunboat_model.keys() if key.split('.')[0] in layers]
            for key in layer_keys:
                self.brain.state_dict()[key].copy_(gunboat_model[key])

        self.brain.to(device)

    async def get_orders(self, game, power_name):
        if game.phase == 'SPRING 1901 MOVEMENT':
            game.add_on_game_message_received(
                notification_callback=lambda x, y: asyncio.create_task(self.reply_press(x, y)))

        start_time = time.time()

        board_state = torch.Tensor(get_board_state(game.get_state())).to(device)
        prev_orders = torch.Tensor(get_last_phase_orders(game)).to(device)

        await self.send_press(game, power_name, board_state, prev_orders)

        while not time.time() - start_time >= self.press_time:
            time.sleep(1)

        orderable_locs = game.get_orderable_locations()

        dist, _ = self.brain(board_state, prev_orders, self.msg_logs[power_name], [power_name], orderable_locs)

        actions = select_orders(dist[power_name], game, power_name, orderable_locs)

        return [ix_to_order(ix) for ix in actions]

    async def send_press(self, game, power_name, board_state, prev_orders):
        msg_dist = self.brain.forward_msgs(board_state, prev_orders, self.msg_logs[power_name])
        msg_dist = torch.sigmoid(msg_dist)
        msg_dist = filter_messages(msg_dist, power_name, game.get_units(power_name))
        msg_ixs = msg_dist.ge(0.5).nonzero()

        for msg_ix in msg_ixs:
            self.add_message(msg_ix, power_name)
            await send_message(game, power_name, msg_ix)

    async def reply_press(self, game, msg_obj):
        power_name = msg_obj.message.recipient
        board_state = torch.Tensor(get_board_state(game.get_state())).to(device)
        prev_orders = torch.Tensor(get_last_phase_orders(game)).to(device)
        message_text = msg_obj.message.message
        reply_power = msg_obj.message.sender

        messages = split_DMZ(message_text)

        for received_message in messages:
            if is_daide_msg_reply(received_message):
                msg_ix, answered_msg_ix = get_msg_ixs_from_daide_reply(received_message)
                if msg_ix:
                    self.add_message(msg_ix, power_name, last_message_ix=answered_msg_ix)
            else:
                last_msg_ix = get_daide_msg_ix(received_message)

                self.add_message(get_daide_msg_ix(received_message), power_name)

                msg_dist = self.brain.forward_answer(board_state, prev_orders, self.msg_logs[power_name])
                msg_dist = F.softmax(msg_dist, dim=0)
                msg_ix = torch.multinomial(msg_dist, 1)

                if msg_ix != 2:
                    self.add_message(msg_ix, power_name, last_msg_ix)
                    await send_message(game, power_name, msg_ix, last_message=received_message, reply_power=reply_power)

        await self.send_press(game, power_name, board_state, prev_orders)

    def add_message(self, msg_ix, power_name, last_message_ix=None):
        if last_message_ix:
            msg_ix = (msg_ix + 1) * len(MESSAGE_LIST) + last_message_ix
        msg_embed = self.brain.msg_embedding(torch.LongTensor([msg_ix]).to(device))
        self.msg_logs[power_name] = torch.cat([self.msg_logs[power_name][1:], msg_embed])


class Brain(nn.Module):
    def __init__(self, state_size=LOC_VECTOR_LENGTH + ORDER_SIZE, embed_size=224, msg_embed_size=100,
                 transformer_layers=10,
                 transformer_heads=8, lstm_size=200, lstm_layers=2, msg_log_size=20):
        super(Brain, self).__init__()

        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers

        self.msg_embedding = nn.Embedding(len(MESSAGE_LIST) * 3, msg_embed_size)

        # Encoder
        self.encoder = Encoder(state_size, embed_size, transformer_layers, transformer_heads)

        # Policy Network
        # LSTM Decoder: encoded state (embed_size) > action probabilities (len(ACTION_LIST))
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(embed_size * 2, lstm_size, num_layers=lstm_layers)
        self.linearPolicy = nn.Linear(lstm_size, len(ACTION_LIST))

        # Value Network
        self.linear1 = nn.Linear(len(LOCATIONS) * embed_size, embed_size)
        self.linear2 = nn.Linear(embed_size, len(ALL_POWERS))

        # messages
        self.msgEmbedLinear = nn.Linear(msg_log_size * msg_embed_size, embed_size)
        self.msgOutputLinear = nn.Linear(len(LOCATIONS) * embed_size + embed_size, len(MESSAGE_LIST))
        self.msgReplyLinear = nn.Linear(len(LOCATIONS) * embed_size + embed_size, len(ANSWER_LIST))

    def init_hidden(self):
        return (torch.zeros(self.lstm_layers, 1, self.lstm_size).to(device),
                torch.zeros(self.lstm_layers, 1, self.lstm_size).to(device))

    def forward(self, x_bo, x_po, msg_logs, powers, locs_by_power):
        x = self.encoder(x_bo, x_po)

        # policy
        dist = {}
        for power in powers:
            if not locs_by_power[power]:
                dist[power] = torch.Tensor([]).to(device)
            else:
                msg_state_embed = F.relu(self.msgEmbedLinear(torch.flatten(msg_logs[power])))
                self.hidden = self.init_hidden()
                locs_ix = [loc_to_ix(loc) for loc in locs_by_power[power]]
                locs_emb = x[locs_ix]
                x_pol, self.hidden = self.lstm(torch.cat([locs_emb, msg_state_embed.repeat(len(locs_ix), 1, 1)], dim=2),
                                               self.hidden)
                x_pol = self.linearPolicy(x_pol)
                dist[power] = torch.reshape(x_pol, (len(locs_ix), -1))

        # value
        x_value = torch.flatten(x)
        x_value = F.relu(self.linear1(x_value))
        value = self.linear2(x_value)

        return dist, value

    def forward_msgs(self, x_bo, x_po, msg_log):
        # calculates the probability of sending each message and sends all above 0.5
        x = self.encoder(x_bo, x_po)
        msg_state_embed = F.relu(self.msgEmbedLinear(torch.flatten(msg_log)))
        x = torch.cat([torch.flatten(x), msg_state_embed])
        x = self.msgOutputLinear(x)

        return torch.flatten(x)

    def forward_answer(self, x_bo, x_po, msg_log):
        # calculates the best response to a message, yes, rej or no answer
        x = self.encoder(x_bo, x_po)
        msg_state_embed = F.relu(self.msgEmbedLinear(torch.flatten(msg_log)))
        x = torch.cat([torch.flatten(x), msg_state_embed])
        x = self.msgReplyLinear(x)

        return x


class Encoder(nn.Module):
    def __init__(self, state_size, embed_size, transformer_layers, transformer_heads):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.state_size = state_size

        self.linear = nn.Linear(self.state_size, embed_size)

        self.positional_bias = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(len(LOCATIONS), 1, embed_size)))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=transformer_heads,
                                                   dim_feedforward=embed_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

    def forward(self, x_bo, x_po):
        x = torch.cat([x_bo, x_po], -1)
        x = self.linear(x)
        x = torch.reshape(x, (-1, 1, self.embed_size))
        x = x + self.positional_bias
        x = self.transformer_encoder(x)
        return x
