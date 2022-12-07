import jsonlines
import torch
from torch import optim as optim, nn as nn

from Player import Player, device
from environment.constants import POWER_ACRONYMS_LIST, ALL_POWERS
from environment.observation_utils import get_board_state, phase_orders_to_rep
from environment.order_utils import order_to_ix


def train_sl(dataset_path, model_path=None, print_ratio=0, save_ratio=1000, output_header='sl_model_DipNet',
             dist_learning_rate=1e-4, value_learning_rate=1e-6,
             embed_size=224, transformer_layers=10, transformer_heads=8, lstm_layers=2):
    def sort_orders(orderable_locs, orders):
        # TODO cleanup funcion, remove useless WAIVES
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

    player = Player(model_path, embed_size=embed_size, transformer_layers=transformer_layers,
                    transformer_heads=transformer_heads, lstm_layers=lstm_layers)

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

    for epoch in range(10):
        game_count = 0
        value_input_count = 0
        dist_input_count = 0
        running_dist_loss = 0.0
        running_value_loss = 0.0
        running_accuracy = 0.0

        for path in dataset_path:
            with jsonlines.open(path) as reader:
                for _ in range(6):
                    reader.read()

                obj = reader.read()
                for _ in range(1000):
                # for obj in reader:
                    game_count += 1

                    # calculate final score for use in value learning
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

                    # remove powers if they end with less than 7 SCs
                    # powers_to_learn = [ALL_POWERS[i] for i, score in enumerate(final_score) if score >= 7]
                    powers_to_learn = ALL_POWERS

                    for phase in obj['phases']:
                        optimizer.zero_grad()

                        board_state = get_board_state(phase['state'])
                        prev_orders = phase_orders_to_rep(last_phase_orders)
                        powers = [power for power, orders in phase['orders'].items() if orders
                                  and power in powers_to_learn]
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
                            value_labels = [score / sum(final_score) for score in final_score]
                        else:
                            value_labels = [0] * 7

                        if dist_labels:
                            dist_loss = criterion(torch.stack(dist_outputs).to(device),
                                                  torch.LongTensor(dist_labels).to(device))
                            dist_loss.backward(retain_graph=True)

                            dist_input_count += 1
                            running_dist_loss += dist_loss.item()
                            # TODO calculate accuracy differently
                            running_accuracy = sum(1 for x, y in zip(dist_outputs, dist_labels) if x.argmax() == y) / len(dist_labels)

                        value_loss = criterion(value_outputs.reshape(1, -1),
                                               torch.Tensor(value_labels).to(device).reshape(1, -1))
                        value_loss.backward()
                        optimizer.step()

                        value_input_count += 1

                        running_value_loss += value_loss.item()

                    if print_ratio != 0 and game_count % print_ratio == 0:
                        print(f'[{epoch + 1}, {game_count}] dist loss: {running_dist_loss / dist_input_count:.3f},'
                              f' value loss: {running_value_loss / value_input_count:.3f},'
                              f' accuracy: {running_accuracy / dist_input_count * 100:.2f}%')
                        running_dist_loss = 0.0
                        running_value_loss = 0.0
                        running_accuracy = 0.0
                        value_input_count = 0
                        dist_input_count = 0

                    if save_ratio != 0 and game_count % save_ratio == 0:
                        torch.save(player.brain.state_dict(), f'models/{output_header}_{epoch + 1}_{game_count}.pth')

        torch.save(player.brain.state_dict(), f'models/sl_model_DipNet_{epoch + 1}_{game_count}_full.pth')
