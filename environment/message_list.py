import itertools
from environment import action_list, constants


def generate_power_combinations(powers):
    """
    Generates combinations of powers

    :param powers: list of powers
    :return: list of combinations of powers
    """
    return sum([list(itertools.combinations(powers, i + 1)) for i in range(1, len(powers) - 1)], [])


def generate_loc_combinations():
    locs = [loc for loc in constants.LOCATIONS if '/' not in loc]
    return list(itertools.combinations(locs, 2)) + locs


def generate_messages(press_level):
    """
    Generates messages based on press level

    :param press_level: level of press
    :return: list of messages
    """
    messages = []
    power_combinations = generate_power_combinations(constants.ALL_POWERS)
    if press_level >= 1:
        messages += [
            *[["PCE", perm] for perm in power_combinations],
            *[["ALY", perm1, perm2] for perm1, perm2 in
              itertools.product(power_combinations, power_combinations + [(power,) for power in constants.ALL_POWERS])
              if set(perm1).isdisjoint(set(perm2))],
        ]

    if press_level >= 2:
        messages += [
            *[["XDO", order] for order in action_list.ACTION_LIST[1:]],
            *[["DMZ", loc, power] for loc, power in
              itertools.product(constants.LOCATIONS,
                                [(power,) for power in constants.ALL_POWERS] +
                                list(itertools.combinations(constants.ALL_POWERS, 2)))
              if '/' not in loc]
        ]
    return messages


PRESS_LEVEL = 2
MESSAGE_LIST = generate_messages(PRESS_LEVEL)

ANSWER_LIST = [
    "YES",
    "REJ",
    None
]
