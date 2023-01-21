import numpy as np
import torch
from diplomacy import Game
from torch.nn import functional as F

from environment.action_list import ACTION_LIST
from environment.constants import *
import environment.observation_utils as observation_utils

ORDER_SIZE = 8

UNIT_TYPE_INDEX = 0
ORDER_TYPE_INDEX = 1
ORDERED_LOC_INDEX = 2
TARGET_LOC_INDEX = 3
EXTRA_LOC_1_INDEX = 4
EXTRA_LOC_INDEXES = [4, 5, 6, 7]


def get_loc_valid_orders(game, loc):
    """
    Given a game object and a location, returns a list of valid orders for the unit located at that location.
    The orders are represented by the index of the order in the ACTION_LIST.
    :param game: game object
    :param loc: a string representing the location
    :return: a list of integers representing valid orders for the unit at the location
    """
    possible_orders = game.get_all_possible_orders()
    orders = [ACTION_LIST.index(order) for order in possible_orders[loc] if order in ACTION_LIST]
    return orders


def loc_to_ix(loc: str) -> int:
    """
    Given a location, returns the index of that location in the LOCATIONS list
    :param loc: a string representing the location
    :return: an integer representing the index of the location in the LOCATIONS list
    """
    return LOCATIONS.index(loc.upper())


def ix_to_loc(ix: int) -> str:
    """
    Given an index, returns the location of that index in the LOCATIONS list
    :param ix: an integer representing an index
    :return: a string representing the location at the given index in the LOCATIONS list
    """
    return LOCATIONS[ix]


def loc_to_rep(loc: str) -> int:
    """
    Given a location, returns its representation, the index in the LOCATIONS list + 1
    :param loc: a string representing the location
    :return: an integer representing the location
    """
    return LOCATIONS.index(loc.upper()) + 1


def rep_to_loc(ix: int) -> str:
    """
    Given an integer, returns the location it represents
    :param ix: an integer representing a location
    :return: a string representing the location associated with the given integer
    """
    return LOCATIONS[ix - 1]


def get_order_type(tokens) -> int:
    """
    Given a list of tokens that form an order,
    this function returns an integer corresponding to the type of the order.

    :param tokens: list of tokens that form an order.
    :return: an integer representing the type of the order.
    """
    order_map = {
        "H": HOLD,
        "-": MOVE,
        "S": SUPPORT_MOVE,
        "C": CONVOY,
        "R": RETREAT_TO,
        "B": BUILD_ARMY,
        "D": DISBAND
    }
    if tokens[0] == 'WAIVE':
        return WAIVE
    elif tokens[2] == '-' and 'VIA' in tokens:
        return CONVOY_TO
    elif tokens[2] == 'B' and tokens[0] == 'F':
        return BUILD_FLEET
    elif tokens[2] == 'S' and '-' not in tokens:
        return SUPPORT_HOLD
    else:
        return order_map.get(tokens[2], None)


def order_to_rep(order: str):
    """
    Given an order as a string, this function returns an array of integers that represents the order in a more
    structured format. The order is represented as follows:
    UNIT TYPE | ORDER TYPE | ORDER LOCATION | TARGET LOCATION | EXTRA LOCATION 1 | EXTRA LOCATION 2 |
    EXTRA LOCATION 3 | EXTRA LOCATION 4

    :param order: a string representing an order.
    :return: a numpy array of size 8 representing the order in a more structured format.
    """

    order_rep = np.zeros(ORDER_SIZE)
    tokens = order.split()

    if tokens[0] in UNIT_TYPES:
        order_rep[UNIT_TYPE_INDEX] = UNIT_TYPES.index(tokens[0])

        order_rep[ORDERED_LOC_INDEX] = loc_to_rep(tokens[1])

        order_type = get_order_type(tokens)
        order_rep[ORDER_TYPE_INDEX] = order_type

        if order_type in [MOVE, CONVOY_TO, RETREAT_TO]:
            order_rep[TARGET_LOC_INDEX] = loc_to_rep(tokens[3])

        elif order_type == SUPPORT_HOLD:
            order_rep[TARGET_LOC_INDEX] = loc_to_rep(tokens[4])

        elif order_type in [SUPPORT_MOVE, CONVOY]:
            order_rep[TARGET_LOC_INDEX] = loc_to_rep(tokens[6])
            order_rep[EXTRA_LOC_1_INDEX] = loc_to_rep(tokens[4])

        elif order_type == CONVOY_TO:
            n_convoys = len(tokens) - 5
            if n_convoys > 4:
                raise ValueError(f"Convoy line longer than 4 {order}")
            for i in range(n_convoys):
                order_rep[EXTRA_LOC_INDEXES[i]] = loc_to_rep(tokens[i+5])

    elif tokens[0] == 'WAIVE':
        order_rep[ORDER_TYPE_INDEX] = WAIVE

    else:
        raise ValueError(f"Unknown order {order}")

    return order_rep


def ix_to_order(order_ix: int) -> str:
    """
    Given an integer representing the index of an order in the list of all possible orders, this function returns the
    corresponding order as a string.

    :param order_ix: an integer representing the index of an order in the list of all possible orders.
    :return: a string representing the order.
    """
    return ACTION_LIST[order_ix]


def order_to_ix(order: str):
    """
    Given an order as a string, this function returns the index of the order in the list of all possible orders.

    :param order: a string representing an order.
    :return: an integer representing the index of the order in the list of all possible orders.
    """
    if 'VIA' in order:
        order = order.split('VIA')[0]
        order += 'VIA'
    if order in ACTION_LIST:
        return ACTION_LIST.index(order)
    else:
        return None


def loc_to_daide_format(loc):
    """
    Given a location as a string, this function returns the location in DAIDE format.
    e.g 'SPA/NC' -> (SPA NCS)

    :param loc: a string representing a location
    :return: a string representing the location in DAIDE format.
    """
    if '/' in loc:
        return f"({loc.split('/')[0]} {loc[4]}CS)"
    else:
        return loc


def order_to_daide_format(order, game, power_name):
    """
    Given an order, the current state of the game and the name of the power that issued the order,
    this function converts the order to the DAIDE format.
    :param order: order as a string.
    :param game: game object.
    :param power_name: name of the power that issued the order.
    :return: order in DAIDE format as a string.
    """
    tokens = order.split()

    if tokens[0] in UNIT_TYPES:
        unit_owner = get_unit_owner(game, ' '.join(tokens[:2]))

        unit_string = f"{unit_owner} {DAIDE_UNIT_TYPES[tokens[0]]} {loc_to_daide_format(tokens[1])}"

        order_type = get_order_type(tokens)

        if order_type == HOLD:
            return f"( {unit_string} ) HLD"
        elif order_type == MOVE:
            return f"( {unit_string} ) MTO {loc_to_daide_format(tokens[3])}"
        elif order_type == CONVOY_TO:
            return f"( {unit_string} ) CTO {loc_to_daide_format(tokens[3])} VIA"
        elif order_type == RETREAT_TO:
            return f"( {unit_string} ) RTO {loc_to_daide_format(tokens[3])}"
        elif order_type == DISBAND:
            return f"( {unit_string} ) DSB"
        elif order_type in [BUILD_ARMY, BUILD_FLEET]:
            return f"( {unit_string} ) BLD"
        elif order_type == REMOVE:
            return f"( {unit_string} ) REM"
        elif order_type in [SUPPORT_HOLD, SUPPORT_MOVE, CONVOY]:
            unit_owner2 = get_unit_owner(game, ' '.join(tokens[3:5]))
            unit_string2 = f"{unit_owner2} {DAIDE_UNIT_TYPES[tokens[3]]} {loc_to_daide_format(tokens[4])}"

            if order_type == SUPPORT_HOLD:
                return f"( {unit_string} ) SUP ( {unit_string2} )"
            elif order_type == SUPPORT_MOVE:
                return f"( {unit_string} ) SUP ( {unit_string2} ) MTO {loc_to_daide_format(tokens[6])}"
            elif order_type == CONVOY:
                return f"( {unit_string} ) CVY ( {unit_string2} ) CTO {loc_to_daide_format(tokens[6])}"
    elif tokens[0] == 'WAIVE':
        return f"{POWER_ACRONYMS[power_name]} WVE"
    else:
        raise ValueError(f"Unknown order {order}")


def get_unit_owner(game, unit):
    """
    Given a game object and a unit, returns the owner of that unit.
    If the unit is not present in the game, raises a ValueError.
    :param game: a game object
    :param unit: a string representing the unit (e.g. 'F ITA' for the Italian Fleet)
    :return: a string representing the owner of the unit (e.g. 'ITA')
    """
    unit_owner_dict = {unit: power for power, units in game.get_state()['units'].items() for unit in units}

    try:
        unit_owner = POWER_ACRONYMS[unit_owner_dict[unit]]
    except KeyError:
        raise ValueError(f"No unit {' '.join(unit)}")

    return unit_owner


def filter_orders(dist, game, power_name, orderable_locs):
    """
   Given a distribution of probabilities of playing each order, a game object, the name of a power,
   and a dictionary of orderable locations, removes invalid orders from the distribution.
   :param dist: a torch tensor representing a distribution of probabilities for all orders
   :param game: a game object
   :param power_name: name of a power
   :param orderable_locs: a dictionary of orderable locations for each power
   :return: a torch tensor representing the filtered distribution of probabilities for all orders
   """
    dist_clone = dist.clone().detach()
    order_mask = torch.ones_like(dist_clone, dtype=torch.bool)
    for i, loc in enumerate(orderable_locs[power_name]):
        order_mask[i, get_loc_valid_orders(game, loc)] = False

    dist_clone = dist_clone.masked_fill(order_mask, value=-torch.inf)

    return dist_clone


def select_orders(dist, game, power_name, orderable_locs):
    """
    Given a distribution of probabilities of playing each order, a game object, the name of a power,
    and a dictionary of orderable locations, samples one saction from the distribution.
    :param dist: a torch tensor representing a distribution of probabilities for all orders
    :param game: a game object
    :param power_name: name of a power
    :param orderable_locs: a dictionary of orderable locations for each power
    :return: a list of indices of the sampled orders
    """
    dist_clone = filter_orders(dist, game, power_name, orderable_locs)

    state = game.get_state()

    n_builds = abs(state['builds'][power_name]['count'])

    if n_builds > 0:
        dist_clone[:, 0] = - torch.inf

        dist_clone = F.softmax(dist_clone.reshape(-1), dim=0)

        actions = [ix % len(dist[0]) for ix in torch.multinomial(dist_clone, n_builds)]
    else:
        dist_clone = F.softmax(dist_clone, dim=1)

        actions = torch.multinomial(dist_clone, 1)

    return actions


def get_max_orders(dist, game, power_name, orderable_locs):
    """
        Given a distribution of probabilities of playing each order, a game object, the name of a power,
        and a dictionary of orderable locations,
        this function filters out invalid orders and returns the indices of the highest probability orders.

        :param dist: a torch tensor representing a distribution of probabilities for all orders
        :param game: a game object
        :param power_name: name of a power
        :param orderable_locs: a dictionary of orderable locations for each power
        :return: List of indices of the highest probability orders
        """
    dist_clone = filter_orders(dist, game, power_name, orderable_locs)

    state = game.get_state()

    n_builds = abs(state['builds'][power_name]['count'])

    if n_builds > 0:
        dist_clone[:, 0] = - torch.inf

        dist_clone = F.softmax(dist_clone.reshape(-1), dim=0)

        actions = [ix % len(dist[0]) for ix in torch.topk(dist_clone, n_builds).indices]
    else:
        dist_clone = F.softmax(dist_clone, dim=1)

        actions = torch.argmax(dist_clone, dim=1)

    return actions
