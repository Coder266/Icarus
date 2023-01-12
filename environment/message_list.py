import itertools

from environment.action_list import ACTION_LIST
from environment.constants import ALL_POWERS, LOCATIONS

POWERS_PERMUTATIONS = sum([list(itertools.combinations(ALL_POWERS, i + 1)) for i in range(1, len(ALL_POWERS) - 1)], [])

PRESS_LEVEL = 1

MESSAGE_LIST = []

ANSWER_LIST = [
    "YES",
    "REJ",
    None
]

if PRESS_LEVEL >= 1:
    MESSAGE_LIST += [
        *[["PCE", perm] for perm in POWERS_PERMUTATIONS],
        *[["ALY", perm1, perm2] for perm1, perm2 in
          itertools.product(POWERS_PERMUTATIONS, POWERS_PERMUTATIONS + [(power,) for power in ALL_POWERS])
          if set(perm1).isdisjoint(set(perm2))],
    ]

if PRESS_LEVEL >= 2:
    MESSAGE_LIST += [
        *[["XDO", order] for order in ACTION_LIST],
        *[["DMZ", loc] for loc in LOCATIONS]
    ]
