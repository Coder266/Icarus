import random

import jsonlines
import torch
from diplomacy import Game
from torch import optim as optim, nn as nn
from torchmetrics.classification import BinaryF1Score

from environment.message_list import MESSAGE_LIST, ANSWER_LIST
from environment.message_utils import get_daide_msg_ix, filter_messages, is_daide_msg_reply, \
    get_msg_ixs_from_daide_reply, split_Albert_DMZs
from players.MessagePlayer import MessagePlayer, device
from environment.constants import POWER_ACRONYMS_LIST, ALL_POWERS
from environment.observation_utils import get_board_state, phase_orders_to_rep
from environment.order_utils import order_to_ix, get_max_orders
from environment.action_list import ACTION_LIST
import logging
import sys


def train_msg_sl(dataset_path, model_path=None, gunboat_model_path=None, print_ratio=0, save_ratio=1000,
                 output_header='sl',
                 log_file=None, dist_learning_rate=1e-4, validation_size=20,
                 embed_size=224, msg_embed_size=100, transformer_layers=5, transformer_heads=8, lstm_size=200,
                 lstm_layers=2, msg_log_size=20, restore_game=None, restore_epoch=None):
    if model_path and gunboat_model_path:
        raise ValueError("Received model_path and gunboat_model_path, please choose only one type of model to"
                         " initialize the network")
    elif not model_path and not gunboat_model_path:
        raise ValueError("Didn't receive model_path or gunboat_model_path, please input one type of model to"
                         " initialize the network")

    # Logging
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(filename=log_file))

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, handlers=handlers)

    # Create player and set parameters
    player = MessagePlayer(model_path=model_path, gunboat_model_path=gunboat_model_path, embed_size=embed_size,
                           msg_embed_size=msg_embed_size, transformer_layers=transformer_layers,
                           transformer_heads=transformer_heads, lstm_size=lstm_size, lstm_layers=lstm_layers,
                           msg_log_size=msg_log_size)

    player.brain.train()

    # Freeze value network (not dependant on messages)
    for layer in [player.brain.linear1, player.brain.linear2]:
        for params in layer.parameters():
            params.requires_grad = False

    optimizer = optim.Adam(
        [
            {"params": player.brain.msg_embedding.parameters()},
            {"params": player.brain.encoder.parameters()},
            {"params": player.brain.lstm.parameters()},
            {"params": player.brain.linearPolicy.parameters()},
            {"params": player.brain.msgEmbedLinear.parameters()},
            {"params": player.brain.msgOutputLinear.parameters()},
            {"params": player.brain.msgReplyLinear.parameters()}
        ],
        lr=dist_learning_rate
    )

    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    num_games = sum(1 for _ in open(dataset_path))

    if restore_epoch:
        start_epoch = restore_epoch - 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, 500):
        game_count = 0
        dist_input_count = 0
        msg_input_count = 0
        running_dist_loss = 0.0
        running_msg_loss = 0.0
        running_total_accuracy = 0.0
        running_power_accuracy = 0.0
        running_msg_score = 0.0

        lines = []
        with jsonlines.open(dataset_path) as reader:
            for obj in reader:
                lines.append(obj)

        random.shuffle(lines)

        # with jsonlines.open(dataset_path) as reader:
        for obj in lines:
            game_count += 1
            if epoch == start_epoch and restore_game and game_count <= restore_game:
                continue

            validate = game_count > (num_games - validation_size)

            if game_count == num_games - validation_size + 1:
                logging.info(f"Calculating accuracy using the validation set (last {validation_size} games)...")

            last_phase = obj['game']['phases'][-1]
            centers = [len(last_phase['state']['centers'][power])
                       if power in last_phase['state']['centers']
                       else 0 for power in ALL_POWERS]

            if sum(centers) <= 0:
                logging.warning(f'Skipping game {game_count} because the total centers are {sum(centers)}')
                continue

            # remove powers if they end with less than 7 SCs, or they're not played by Albert
            powers_to_learn = [ALL_POWERS[i] for i, score in enumerate(centers) if score >= 7
                               and ALL_POWERS[i] in obj['albert_powers']]

            if not powers_to_learn:
                logging.info(f"Skipping game because none of the Albert powers reached 7 SCs")
                continue

            msg_logs = {power: torch.zeros([msg_log_size, msg_embed_size]).to(device) for power in powers_to_learn}

            last_phase_orders = []
            for phase in obj['game']['phases']:
                msg_logs = {power: msg_log.detach() for power, msg_log in msg_logs.items()}
                board_state = get_board_state(phase['state'])
                prev_orders = phase_orders_to_rep(last_phase_orders)

                for power in powers_to_learn:
                    sent_msg_log = []
                    messages = split_Albert_DMZs(phase['messages'])
                    messages = get_power_msgs(messages, power)

                    while messages:
                        if messages[0]['msg_type'] == 'sent':
                            sent_msg_log.append(messages.pop(0))

                        elif messages[0]['msg_type'] == 'received':
                            # train and log accumulated sent messages
                            if sent_msg_log:
                                running_msg_loss, running_msg_score, msg_input_count, msg_logs[power] = \
                                    train_and_log_sent_msgs(board_state, prev_orders, player.brain, power,
                                                            phase['state']['units'][power], msg_logs[power],
                                                            sent_msg_log,
                                                            bce_loss, validate, msg_input_count,
                                                            running_msg_loss, running_msg_score)

                            # add received message to log
                            received_msg = messages.pop(0)
                            if received_msg['is_reply']:
                                msg_logs[power] = add_to_log(msg_logs[power], player.brain,
                                                             [received_msg['msg_ix']],
                                                             received_msg['answered_msg_ix'])
                            else:
                                msg_logs[power] = add_to_log(msg_logs[power], player.brain,
                                                             [received_msg['msg_ix']])

                                # find possible reply
                                reply = None
                                for i, msg in enumerate(messages):
                                    if msg['msg_type'] == 'reply' and \
                                            received_msg['msg_ix'] == msg['answered_msg_ix']:
                                        reply = messages.pop(i)
                                        break

                                # train and log reply or lack of
                                reply_ix = reply['msg_ix'] if reply else ANSWER_LIST.index(None)
                                running_msg_loss, running_msg_score, msg_input_count, msg_logs[power] = \
                                    train_and_log_reply(board_state, prev_orders, player.brain, msg_logs[power],
                                                        reply_ix, received_msg['msg_ix'], ce_loss, validate,
                                                        msg_input_count, running_msg_loss, running_msg_score)

                        elif messages[0]['msg_type'] == 'reply':
                            raise ValueError(f"Message {messages[0]} replying to unsent message")
                        else:
                            raise ValueError(f"Invalid message {messages[0]}")

                    if sent_msg_log:
                        running_msg_loss, running_msg_score, msg_input_count, msg_logs[power] = \
                            train_and_log_sent_msgs(board_state, prev_orders, player.brain, power,
                                                    phase['state']['units'][power], msg_logs[power],
                                                    sent_msg_log,
                                                    bce_loss, validate, msg_input_count,
                                                    running_msg_loss, running_msg_score)

                powers = [power for power, orders in phase['orders'].items() if orders
                          and power in powers_to_learn]
                orders = {power: [order for order in orders if order != "WAIVE" and order in ACTION_LIST]
                          for power, orders in phase['orders'].items() if power in powers}
                orderable_locs = {power: [order.split()[1] for order in orders]
                                  for power, orders in orders.items()}

                last_phase_orders = orders

                dist, _ = player.brain(torch.Tensor(board_state).to(device),
                                       torch.Tensor(prev_orders).to(device),
                                       msg_logs,
                                       powers,
                                       orderable_locs)

                # policy network update
                dist_outputs = [probs for power in powers for probs in dist[power]]
                dist_labels = [order_to_ix(order) for power in powers for order in orders[power]]

                if dist_labels:
                    if not validate:
                        dist_loss = ce_loss(torch.stack(dist_outputs).to(device),
                                              torch.LongTensor(dist_labels).to(device))
                        dist_loss.backward(retain_graph=True)

                        running_dist_loss += dist_loss.item()
                    # metrics
                    dist_input_count += 1

                    total_accuracy, power_accuracy = calculate_accuracy(phase['state'],
                                                                        powers, dist, orders, orderable_locs)
                    running_total_accuracy += total_accuracy
                    running_power_accuracy += power_accuracy

                # value network update
                if not validate:
                    optimizer.step()
                    optimizer.zero_grad()

            if print_ratio != 0 and game_count % print_ratio == 0 and not validate:
                logging.info(f'[{epoch + 1}, {game_count}] dist loss: {running_dist_loss / dist_input_count:.3f},'
                             f' total accuracy: {running_total_accuracy / dist_input_count * 100:.2f}%,'
                             f' power accuracy: {running_power_accuracy / dist_input_count * 100:.2f}%,'
                             f' message loss: {running_msg_loss / msg_input_count:.3f},'
                             f' message f1-score: {running_msg_score / msg_input_count * 100:.2f}%')

                running_dist_loss = 0.0
                running_total_accuracy = 0.0
                running_power_accuracy = 0.0
                running_msg_loss = 0.0
                running_msg_score = 0.0
                dist_input_count = 0
                msg_input_count = 0

            if save_ratio != 0 and game_count % save_ratio == 0:
                torch.save(player.brain.state_dict(), f'models/{output_header}_{epoch + 1}_{game_count}.pth')

        logging.info(f"Validation set accuracy for epoch {epoch + 1}:\n"
                     f' total accuracy: {running_total_accuracy / dist_input_count * 100:.2f}%\n'
                     f' power accuracy: {running_power_accuracy / dist_input_count * 100:.2f}%\n'
                     f' message accuracy: {running_msg_score / msg_input_count * 100:.2f}%\n')

        if save_ratio != 0:
            torch.save(player.brain.state_dict(), f'models/{output_header}_{epoch + 1}_{game_count}_full.pth')


def get_power_msgs(messages, power):
    power_msgs = []
    for msg in messages:
        if msg['recipient'] == power:
            if is_daide_msg_reply(msg['message']):
                msg_ix, answered_msg_ix = get_msg_ixs_from_daide_reply(msg['message'])
                power_msgs.append({'msg_type': 'received',
                                   'msg': msg['message'],
                                   'msg_ix': msg_ix,
                                   'answered_msg_ix': answered_msg_ix,
                                   'is_reply': True})
            else:
                power_msgs.append({'msg_type': 'received',
                                   'msg': msg['message'],
                                   'msg_ix': get_daide_msg_ix(msg['message']),
                                   'is_reply': False})

        elif msg['sender'] == power:
            if is_daide_msg_reply(msg['message']):
                msg_ix, answered_msg_ix = get_msg_ixs_from_daide_reply(msg['message'])
                power_msgs.append({'msg_type': 'reply',
                                   'msg': msg['message'],
                                   'msg_ix': msg_ix,
                                   'answered_msg_ix': answered_msg_ix})
            else:
                power_msgs.append({'msg_type': 'sent',
                                   'msg': msg['message'],
                                   'msg_ix': get_daide_msg_ix(msg['message'])})

    return power_msgs


def add_to_log(msg_log, model, msg_ixs, last_message_ix=None):
    if last_message_ix:
        msg_ixs[0] = (msg_ixs[0] + 1) * len(MESSAGE_LIST) + last_message_ix

    msg_embeds = model.msg_embedding(torch.LongTensor(msg_ixs).to(device))
    msg_log = torch.cat([msg_log[len(msg_ixs):], msg_embeds])
    return msg_log


def train_and_log_sent_msgs(board_state, prev_orders, model, power, units, msg_log, sent_msg_log, loss_fn, validate,
                            msg_input_count, running_msg_loss, running_msg_score):
    msg_ixs = [get_daide_msg_ix(msg['msg']) for msg in sent_msg_log]
    msg_loss, msg_score = train_msgs(board_state, prev_orders, model, power, units, msg_log, msg_ixs,
                                     loss_fn, validate)

    msg_input_count += 1
    if msg_loss:
        running_msg_loss += msg_loss.item()
    running_msg_score += msg_score

    msg_log = add_to_log(msg_log, model, msg_ixs)

    return running_msg_loss, running_msg_score, msg_input_count, msg_log


def train_msgs(board_state, prev_orders, model, power, units, msg_log, msg_ixs, loss_fn, validate):
    # generate messages
    gen_msg_dist = model.forward_msgs(torch.Tensor(board_state).to(device),
                                      torch.Tensor(prev_orders).to(device),
                                      msg_log)

    # compare generated to real sent messages
    real_msg_dist = torch.zeros_like(gen_msg_dist).to(device)

    real_msg_dist[msg_ixs] = 1.0
    msg_loss = None

    if not validate:
        msg_loss = loss_fn(gen_msg_dist, real_msg_dist)
        msg_loss.backward(retain_graph=True)

    # metrics
    gen_msg_dist = torch.sigmoid(gen_msg_dist)
    gen_msg_dist = filter_messages(gen_msg_dist, power, units)
    gen_msg_dist = gen_msg_dist.ge(0.2)

    f1 = BinaryF1Score().to(device)
    msg_score = f1(gen_msg_dist.long(), real_msg_dist)

    return msg_loss, msg_score


def train_and_log_reply(board_state, prev_orders, model, msg_log, reply_ix, answered_msg_ix, loss_fn, validate,
                        msg_input_count, running_msg_loss, running_msg_score):
    msg_loss, msg_score = train_reply(board_state, prev_orders, model, msg_log, reply_ix, loss_fn, validate)

    msg_input_count += 1
    if msg_loss:
        running_msg_loss += msg_loss.item()
    running_msg_score += msg_score

    if ANSWER_LIST[reply_ix] is not None:
        msg_log = add_to_log(msg_log, model, [reply_ix], last_message_ix=answered_msg_ix)

    return running_msg_loss, running_msg_score, msg_input_count, msg_log


def train_reply(board_state, prev_orders, model, msg_log, real_reply_ix, loss_fn, validate):
    # generate messages
    gen_msg_dist = model.forward_answer(torch.Tensor(board_state).to(device),
                                        torch.Tensor(prev_orders).to(device),
                                        msg_log)

    # compare generated to real sent messages
    real_msg_dist = torch.zeros_like(gen_msg_dist).to(device)

    real_msg_dist[real_reply_ix] = 1.0
    msg_loss = None

    if not validate:
        msg_loss = loss_fn(gen_msg_dist.reshape(1, -1), torch.LongTensor([real_reply_ix]).to(device))
        msg_loss.backward(retain_graph=True)

    # metrics
    msg_dist = torch.zeros_like(gen_msg_dist).to(device)
    msg_dist[torch.argmax(gen_msg_dist)] = 1.0

    f1 = BinaryF1Score().to(device)
    msg_score = f1(msg_dist, real_msg_dist)

    return msg_loss, msg_score


def calculate_accuracy(state, powers, dist, orders, orderable_locs):
    game = Game()
    game.set_state(state)
    count = 0
    total_accuracy = 0
    power_accuracy = {}
    for power in powers:
        if orders[power]:
            power_outputs = get_max_orders(dist[power], game, power, orderable_locs)

            # orders to labels
            power_labels = [order_to_ix(order) for order in orders[power]]

            n_builds = abs(state['builds'][power]['count'])

            if n_builds > 0:
                power_outputs = [tensor.item() for tensor in power_outputs]
                accuracy = len(set(power_outputs).intersection(set(power_labels)))
                total_accuracy += accuracy
                power_accuracy[power] = accuracy == len(power_labels)
            else:
                if len(power_labels) > 0:
                    accuracy = sum(1 for x, y in zip(power_outputs, power_labels) if x == y)
                    total_accuracy += accuracy
                    power_accuracy[power] = accuracy == len(power_labels)

            count += len(power_labels)

    return total_accuracy / count, sum(power_accuracy.values()) / len(power_accuracy)
