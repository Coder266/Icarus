import jsonlines
import torch
from diplomacy import Game
from torch import optim as optim, nn as nn

from players.Player import Player, device
from environment.constants import POWER_ACRONYMS_LIST, ALL_POWERS
from environment.observation_utils import get_board_state, phase_orders_to_rep
from environment.order_utils import order_to_ix, get_max_orders
from environment.action_list import ACTION_LIST
import logging
import sys


def train_sl(dataset_path, model_path=None, print_ratio=0, save_ratio=1000, output_header='sl_model_DipNet',
             log_file=None, dist_learning_rate=1e-4, value_learning_rate=1e-6, validation_size=200,
             embed_size=224, transformer_layers=10, transformer_heads=8, lstm_size=200, lstm_layers=2,
             restore_game=None, restore_epoch=None):

    # Logging
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(filename=log_file))

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, handlers=handlers)

    # Create player and set parameters
    player = Player(model_path, embed_size=embed_size, transformer_layers=transformer_layers,
                    transformer_heads=transformer_heads, lstm_size=lstm_size, lstm_layers=lstm_layers)

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

    num_games = sum(1 for _ in open(dataset_path))

    if restore_epoch:
        start_epoch = restore_epoch - 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, 500):
        game_count = 0
        value_input_count = 0
        dist_input_count = 0
        running_dist_loss = 0.0
        running_value_loss = 0.0
        running_total_accuracy = 0.0
        running_power_accuracy = 0.0

        with jsonlines.open(dataset_path) as reader:
            for obj in reader:
                game_count += 1
                if epoch == start_epoch and restore_game and game_count <= restore_game:
                    continue

                validate = game_count > (num_games - validation_size)

                if game_count == num_games - validation_size + 1:
                    logging.info(f"Calculating accuracy using the validation set (last {validation_size} games)...")

                # calculate final score for use in value learning
                note = obj['phases'][-1]['state']['note'].split(': ')
                note[1] = note[1].split(', ')

                if note[0] == 'Victory by':
                    final_score = [int(power in note[1]) * (34 / len(note[1])) for power in POWER_ACRONYMS_LIST]
                else:
                    last_phase = obj['phases'][-1]
                    # final_score should be
                    # final_score = [score/sum(final_score) * 34 for score in final_score]
                    # but it is not needed here since we only need the proportion
                    final_score = [len(last_phase['state']['centers'][power])
                                   if power in last_phase['state']['centers']
                                   else 0 for power in ALL_POWERS]

                if sum(final_score) <= 0:
                    logging.warning(f'Skipping game {game_count} because final score is {sum(final_score)}')
                    continue

                value_labels = [score / sum(final_score) for score in final_score]

                # remove powers if they end with less than 7 SCs
                powers_to_learn = [ALL_POWERS[i] for i, score in enumerate(final_score) if score >= 7]

                last_phase_orders = []
                for phase in obj['phases']:
                    board_state = get_board_state(phase['state'])
                    prev_orders = phase_orders_to_rep(last_phase_orders)
                    powers = [power for power, orders in phase['orders'].items() if orders
                              and power in powers_to_learn]
                    orders = {power: [order for order in orders if order != "WAIVE" and order in ACTION_LIST]
                              for power, orders in phase['orders'].items() if power in powers}
                    orderable_locs = {power: [order.split()[1] for order in orders]
                                      for power, orders in orders.items()}

                    last_phase_orders = orders

                    dist, value_outputs = player.brain(torch.Tensor(board_state).to(device),
                                                       torch.Tensor(prev_orders).to(device),
                                                       powers,
                                                       orderable_locs)

                    # policy network update
                    dist_outputs = [probs for power in powers for probs in dist[power]]
                    dist_labels = [order_to_ix(order) for power in powers for order in orders[power]]

                    if dist_labels:
                        if not validate:
                            dist_loss = criterion(torch.stack(dist_outputs).to(device),
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
                        value_loss = criterion(value_outputs.reshape(1, -1),
                                               torch.Tensor(value_labels).to(device).reshape(1, -1))
                        value_loss.backward()

                        running_value_loss += value_loss.item()

                        optimizer.step()
                        optimizer.zero_grad()

                    # metrics
                    value_input_count += 1

                if print_ratio != 0 and game_count % print_ratio == 0 and not validate:
                    logging.info(f'[{epoch + 1}, {game_count}] dist loss: {running_dist_loss / dist_input_count:.3f},'
                                 f' value loss: {running_value_loss / value_input_count:.3f},'
                                 f' total accuracy: {running_total_accuracy / dist_input_count * 100:.2f}%,'
                                 f' power accuracy: {running_power_accuracy / dist_input_count * 100:.2f}%')
                    running_dist_loss = 0.0
                    running_value_loss = 0.0
                    running_total_accuracy = 0.0
                    running_power_accuracy = 0.0
                    value_input_count = 0
                    dist_input_count = 0

                if save_ratio != 0 and game_count % save_ratio == 0:
                    torch.save(player.brain.state_dict(), f'models/{output_header}_{epoch + 1}_{game_count}.pth')

        logging.info(f"Validation set accuracy for epoch {epoch + 1}:\n"
                     f' total accuracy: {running_total_accuracy / dist_input_count * 100:.2f}%\n'
                     f' power accuracy: {running_power_accuracy / dist_input_count * 100:.2f}%\n')

        if save_ratio != 0:
            torch.save(player.brain.state_dict(), f'models/{output_header}_{epoch + 1}_{game_count}_full.pth')


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
