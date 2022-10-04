from environment.constants import ALL_POWERS
import matplotlib.pyplot as plt


class StatTracker:
    def __init__(self):
        self.wins = {power: 0 for power in ALL_POWERS}
        self.score = None

    def new_game(self, game):
        self.score = {power: [len(game.get_state()["centers"][power])] for power in ALL_POWERS}

    def update(self, game):
        for power in ALL_POWERS:
            self.score[power].append(
                len(game.get_state()["centers"][power]))

    def end_game(self):
        final_score = {power: self.score[power][-1] for power in ALL_POWERS}
        self.wins[max(final_score, key=final_score.get)] += 1

    def plot_game(self):
        for i, (key, data_list) in enumerate(self.score.items()):
            # shift = (i - len(self.score) / 2) * 0.03
            # shifted_data = [data_point + shift for data_point in data_list]
            plt.plot(range(len(data_list)), data_list, label=key, alpha=0.8)
        plt.yticks(range(19))
        plt.grid()
        plt.legend()
        plt.show()

    def plot_wins(self):
        plt.bar(range(len(self.wins)), list(self.wins.values()), align='center')
        plt.xticks(range(len(self.wins)), list(self.wins.keys()))
        plt.show()

    def print_wins(self):
        print(f'Wins: {self.wins}')
