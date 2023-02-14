import itertools
from environment import action_list, constants


def generate_permutations(powers):
    """
    Generates permutations of powers

    :param powers: list of powers
    :return: list of permutations of powers
    """
    return sum([list(itertools.combinations(powers, i + 1)) for i in range(1, len(powers) - 1)], [])


def generate_messages(press_level):
    """
    Generates messages based on press level

    :param press_level: level of press
    :return: list of messages
    """
    messages = []
    powers_permutations = generate_permutations(constants.ALL_POWERS)
    if press_level >= 1:
        messages += [
            *[["PCE", perm] for perm in powers_permutations],
            *[["ALY", perm1, perm2] for perm1, perm2 in
              itertools.product(powers_permutations, powers_permutations + [(power,) for power in constants.ALL_POWERS])
              if set(perm1).isdisjoint(set(perm2))],
        ]

    if press_level >= 2:
        messages += [
            *[["XDO", order] for order in action_list.ACTION_LIST[1:]],
            *[["DMZ", loc] for loc in constants.LOCATIONS if '/' not in loc]
        ]
    return messages


PRESS_LEVEL = 2
MESSAGE_LIST = generate_messages(PRESS_LEVEL)

ANSWER_LIST = [
    "YES",
    "REJ",
    None
]
