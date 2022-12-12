import numpy as np
import torch
from diplomacy import Game
from torch.nn import functional as F

from environment.action_list import ACTION_LIST
from environment.constants import *
import environment.observation_utils as observation_utils

ORDER_SIZE = 8

# unit type, 0 for army, 1 for fleet
UNIT_TYPE_START = 0
UNIT_TYPE_BITS = 1
ORDER_TYPE_START = 1
ORDER_TYPE_BITS = 4
LOC_BITS = 8
ORDERED_LOC_START = 8
TARGET_LOC_START = 16
# up to 4 extra locations
# support move and convoy each use an extra location
# convoy_to is limited to 4 convoys
EXTRA_LOC_1_START = 24
EXTRA_LOC_2_START = 32
EXTRA_LOC_3_START = 40
EXTRA_LOC_4_START = 48

UNIT_TYPE_INDEX = 0
ORDER_TYPE_INDEX = 1
ORDERED_LOC_INDEX = 2
TARGET_LOC_INDEX = 3
EXTRA_LOC_1_INDEX = 4
EXTRA_LOC_2_INDEX = 5
EXTRA_LOC_3_INDEX = 6
EXTRA_LOC_4_INDEX = 7


def get_valid_orders(game, power):
    possible_orders = game.get_all_possible_orders()
    return [ACTION_LIST.index(order) for loc in game.get_orderable_locations(power) for order in possible_orders[loc]]


def get_loc_valid_orders(game, loc):
    possible_orders = game.get_all_possible_orders()
    orders = [ACTION_LIST.index(order) for order in possible_orders[loc] if order in ACTION_LIST]
    return orders


def loc_to_ix(loc: str) -> int:
    return LOCATIONS.index(loc.upper())


def loc_to_rep(loc: str) -> int:
    return LOCATIONS.index(loc.upper()) + 1


def rep_to_loc(ix: int) -> str:
    return LOCATIONS[ix - 1]


def ix_to_loc(ix: int) -> str:
    return LOCATIONS[ix]


def get_order_type(tokens) -> int:
    if tokens[0] == 'WAIVE':
        return WAIVE

    if tokens[2] == 'H':
        return HOLD

    elif tokens[2] == '-':
        if 'VIA' not in tokens:
            return MOVE
        else:
            return CONVOY_TO

    elif tokens[2] == 'S':
        if '-' not in tokens:
            return SUPPORT_HOLD
        else:
            return SUPPORT_MOVE

    elif tokens[2] == 'C':
        return CONVOY

    elif tokens[2] == 'R':
        return RETREAT_TO

    elif tokens[2] == 'B':
        if tokens[0] == 'A':
            return BUILD_ARMY
        elif tokens[0] == 'F':
            return BUILD_FLEET
        else:
            raise ValueError(f"Unknown unit type for build order {tokens[0]}")

    elif tokens[2] == 'D':
        return DISBAND

    else:
        raise ValueError(f"Unknown order type {tokens[2]}")


def order_to_id(order: str) -> int:
    # UNIT TYPE (1 bit) | ORDER TYPE (4 bits) | ORDER LOCATION (8 bits) | TARGET LOCATION (8 bits) |
    # EXTRA LOCATION 1 (8 bits) | EXTRA LOCATION 2 (8 bits) | EXTRA LOCATION 3 (8 bits) | EXTRA LOCATION 4 (8 bits)

    order_id = 0
    tokens = order.split()

    # tokens[0] is unit
    if tokens[0] in UNIT_TYPES:
        order_id |= UNIT_TYPES.index(tokens[0]) << UNIT_TYPE_START

        order_id |= loc_to_rep(tokens[1]) << ORDERED_LOC_START

        order_type = get_order_type(tokens)
        order_id |= order_type << ORDER_TYPE_START

        if order_type in [MOVE, CONVOY_TO, RETREAT_TO]:
            order_id |= loc_to_rep(tokens[3]) << TARGET_LOC_START

        elif order_type == SUPPORT_HOLD:
            order_id |= loc_to_rep(tokens[4]) << TARGET_LOC_START

        elif order_type in [SUPPORT_MOVE, CONVOY]:
            order_id |= loc_to_rep(tokens[6]) << TARGET_LOC_START
            order_id |= loc_to_rep(tokens[4]) << EXTRA_LOC_1_START

        elif order_type == CONVOY_TO:
            n_convoys = len(tokens) - 5
            if n_convoys > 4:
                raise ValueError(f"Convoy line longer than 4 {order}")
            if n_convoys > 0:
                order_id |= loc_to_rep(tokens[5]) << EXTRA_LOC_1_START
            if n_convoys > 1:
                order_id |= loc_to_rep(tokens[6]) << EXTRA_LOC_2_START
            if n_convoys > 2:
                order_id |= loc_to_rep(tokens[7]) << EXTRA_LOC_3_START
            if n_convoys > 3:
                order_id |= loc_to_rep(tokens[8]) << EXTRA_LOC_4_START

    elif tokens[0] == 'WAIVE':
        order_id |= WAIVE << ORDER_TYPE_START

    else:
        raise ValueError(f"Unknown order {order}")

    return order_id


def order_to_rep(order: str):
    # UNIT TYPE | ORDER TYPE | ORDER LOCATION | TARGET LOCATION |
    # EXTRA LOCATION 1 | EXTRA LOCATION 2 | EXTRA LOCATION 3 | EXTRA LOCATION 4

    order_rep = np.zeros(ORDER_SIZE)
    tokens = order.split()

    # tokens[0] is unit
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
            if n_convoys > 0:
                order_rep[EXTRA_LOC_1_INDEX] = loc_to_rep(tokens[5])
            if n_convoys > 1:
                order_rep[EXTRA_LOC_2_INDEX] = loc_to_rep(tokens[6])
            if n_convoys > 2:
                order_rep[EXTRA_LOC_3_INDEX] = loc_to_rep(tokens[7])
            if n_convoys > 3:
                order_rep[EXTRA_LOC_4_INDEX] = loc_to_rep(tokens[8])

    elif tokens[0] == 'WAIVE':
        order_rep[ORDER_TYPE_INDEX] = WAIVE

    else:
        raise ValueError(f"Unknown order {order}")

    return order_rep


def id_to_order(order_id: int, game: Game) -> str:
    unit_type = bits_between(order_id, UNIT_TYPE_START, UNIT_TYPE_START + UNIT_TYPE_BITS)
    order_type = bits_between(order_id, ORDER_TYPE_START, ORDER_TYPE_START + ORDER_TYPE_BITS)
    order_loc_ix = bits_between(order_id, ORDERED_LOC_START, ORDERED_LOC_START + LOC_BITS)
    target_loc_ix = bits_between(order_id, TARGET_LOC_START, TARGET_LOC_START + LOC_BITS)
    extra_loc1_ix = bits_between(order_id, EXTRA_LOC_1_START, EXTRA_LOC_1_START + LOC_BITS)
    extra_loc2_ix = bits_between(order_id, EXTRA_LOC_2_START, EXTRA_LOC_2_START + LOC_BITS)
    extra_loc3_ix = bits_between(order_id, EXTRA_LOC_3_START, EXTRA_LOC_3_START + LOC_BITS)
    extra_loc4_ix = bits_between(order_id, EXTRA_LOC_4_START, EXTRA_LOC_4_START + LOC_BITS)

    if order_type == WAIVE:
        return 'WAIVE'
    else:
        order = []
        # unit type and location
        order += [UNIT_TYPES[unit_type], rep_to_loc(order_loc_ix)]

        # order code
        if order_type == HOLD:
            order += ['H']
        elif order_type in [MOVE, CONVOY_TO]:
            order += ['-']
        elif order_type in [SUPPORT_HOLD, SUPPORT_MOVE]:
            order += ['S']
        elif order_type == CONVOY:
            order += ['C']
        elif order_type == RETREAT_TO:
            order += ['R']
        elif order_type == DISBAND:
            order += ['D']
        elif order_type in [BUILD_ARMY, BUILD_FLEET]:
            order += ['B']
        elif order_type == REMOVE:
            order += ['R']
        else:
            # order += [' ']
            raise ValueError(f'Unknown order type {order_type}')

        # locations
        if order_type in [MOVE, CONVOY_TO, RETREAT_TO]:
            order += [rep_to_loc(target_loc_ix)]
        elif order_type == SUPPORT_HOLD:
            order += [observation_utils.get_unit_type_in_loc(rep_to_loc(target_loc_ix), game),
                      rep_to_loc(target_loc_ix)]
        elif order_type in [SUPPORT_MOVE, CONVOY]:
            order += [observation_utils.get_unit_type_in_loc(rep_to_loc(extra_loc1_ix), game),
                      rep_to_loc(extra_loc1_ix),
                      '-',
                      rep_to_loc(target_loc_ix)]

        if order_type == CONVOY_TO:
            order += ['VIA']
            if extra_loc1_ix != 0:
                order += rep_to_loc(extra_loc1_ix)
            if extra_loc2_ix != 0:
                order += rep_to_loc(extra_loc2_ix)
            if extra_loc3_ix != 0:
                order += rep_to_loc(extra_loc3_ix)
            if extra_loc4_ix != 0:
                order += rep_to_loc(extra_loc4_ix)

        return ' '.join(order)


def ix_to_order(order_ix: int) -> str:
    return ACTION_LIST[order_ix]


def order_to_ix(order: str):
    if 'VIA' in order:
        order = order.split('VIA')[0]
        order += 'VIA'
    if order in ACTION_LIST:
        return ACTION_LIST.index(order)
    else:
        return None


def bits_between(number: int, start: int, end: int):
    """Returns bits between positions start and end from number."""
    return number % (1 << end) // (1 << start)


def loc_to_daide_format(loc):
    # 'SPA/NC' -> (SPA NCS)
    if '/' in loc:
        return f"({loc.split('/')[0]} {loc[4]}CS)"
    else:
        return loc


def order_to_daide_format(order, game, power_name):
    tokens = order.split()

    if tokens[0] in UNIT_TYPES:
        unit_owner = get_unit_owner(game, ' '.join(tokens[:2]))

        unit_string = f"{unit_owner} {DAIDE_UNIT_TYPES[tokens[0]]} {loc_to_daide_format(tokens[1])}"

        order_type = get_order_type(tokens)

        if order_type == HOLD:
            return f"({unit_string}) HLD"
        elif order_type == MOVE:
            return f"({unit_string}) MTO {loc_to_daide_format(tokens[3])}"
        elif order_type == CONVOY_TO:
            return f"({unit_string}) CTO {loc_to_daide_format(tokens[3])} VIA"
        elif order_type == RETREAT_TO:
            return f"({unit_string}) RTO {loc_to_daide_format(tokens[3])}"
        elif order_type == DISBAND:
            return f"({unit_string}) DSB"
        elif order_type in [BUILD_ARMY, BUILD_FLEET]:
            return f"({unit_string}) BLD"
        elif order_type == REMOVE:
            return f"({unit_string}) REM"
        elif order_type in [SUPPORT_HOLD, SUPPORT_MOVE, CONVOY]:
            unit_owner2 = get_unit_owner(game, ' '.join(tokens[3:5]))
            unit_string2 = f"{unit_owner2} {DAIDE_UNIT_TYPES[tokens[3]]} {loc_to_daide_format(tokens[4])}"

            if order_type == SUPPORT_HOLD:
                return f"({unit_string}) SUP ({unit_string2})"
            elif order_type == SUPPORT_MOVE:
                return f"({unit_string}) SUP ({unit_string2}) MTO {loc_to_daide_format(tokens[6])}"
            elif order_type == CONVOY:
                return f"({unit_string}) CVY ({unit_string2}) CTO {loc_to_daide_format(tokens[6])}"
    elif tokens[0] == 'WAIVE':
        return f"{POWER_ACRONYMS[power_name]} WVE"
    else:
        raise ValueError(f"Unknown order {order}")


def get_unit_owner(game, unit):
    unit_owner_dict = {unit: power for power, units in game.get_state()['units'].items() for unit in units}

    try:
        unit_owner = POWER_ACRONYMS[unit_owner_dict[unit]]
    except KeyError:
        raise ValueError(f"No unit {' '.join(unit)}")

    return unit_owner


def filter_orders(dist, power_name, game, orderable_locs):
    dist_clone = dist.clone().detach()
    order_mask = torch.ones_like(dist_clone, dtype=torch.bool)
    for i, loc in enumerate(orderable_locs[power_name]):
        order_mask[i, get_loc_valid_orders(game, loc)] = False

    dist_clone = dist_clone.masked_fill(order_mask, value=-torch.inf)

    return dist_clone


def select_orders(dist, power_name, game, orderable_locs):
    dist_clone = filter_orders(dist, power_name, game, orderable_locs)

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


def get_max_orders(dist, power_name, game, orderable_locs):
    dist_clone = filter_orders(dist, power_name, game, orderable_locs)

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
