from diplomacy import Game

from environment.action_list import ACTION_LIST
from environment.constants import *
import environment.observation_utils as observation_utils


ORDER_SIZE = 56

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


def loc_to_ix(loc: str) -> int:
    return LOCATIONS.index(loc.upper()) + 1


def ix_to_loc(ix: int) -> str:
    return LOCATIONS[ix - 1]


def get_order_type(tokens) -> int:
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

        order_id |= loc_to_ix(tokens[1]) << ORDERED_LOC_START

        order_type = get_order_type(tokens)
        order_id |= order_type << ORDER_TYPE_START

        if order_type in [MOVE, CONVOY_TO, RETREAT_TO]:
            order_id |= loc_to_ix(tokens[3]) << TARGET_LOC_START

        elif order_type == SUPPORT_HOLD:
            order_id |= loc_to_ix(tokens[4]) << TARGET_LOC_START

        elif order_type in [SUPPORT_MOVE, CONVOY]:
            order_id |= loc_to_ix(tokens[6]) << TARGET_LOC_START
            order_id |= loc_to_ix(tokens[4]) << EXTRA_LOC_1_START

        elif order_type == CONVOY_TO:
            n_convoys = len(tokens) - 5
            if n_convoys > 4:
                raise ValueError(f"Convoy line longer than 4 {order}")
            if n_convoys > 0:
                order_id |= loc_to_ix(tokens[5]) << EXTRA_LOC_1_START
            if n_convoys > 1:
                order_id |= loc_to_ix(tokens[6]) << EXTRA_LOC_2_START
            if n_convoys > 2:
                order_id |= loc_to_ix(tokens[7]) << EXTRA_LOC_3_START
            if n_convoys > 3:
                order_id |= loc_to_ix(tokens[8]) << EXTRA_LOC_4_START

    elif tokens[0] == 'WAIVE':
        order_id |= WAIVE << ORDER_TYPE_START

    else:
        raise ValueError(f"Unknown order {order}")

    return order_id


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
        order += [UNIT_TYPES[unit_type], ix_to_loc(order_loc_ix)]

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
            order += [ix_to_loc(target_loc_ix)]
        elif order_type == SUPPORT_HOLD:
            order += [observation_utils.get_unit_type_in_loc(ix_to_loc(target_loc_ix), game),
                      ix_to_loc(target_loc_ix)]
        elif order_type in [SUPPORT_MOVE, CONVOY]:
            order += [observation_utils.get_unit_type_in_loc(ix_to_loc(extra_loc1_ix), game),
                      ix_to_loc(extra_loc1_ix),
                      '-',
                      ix_to_loc(target_loc_ix)]

        if order_type == CONVOY_TO:
            order += ['VIA']
            if extra_loc1_ix != 0:
                order += ix_to_loc(extra_loc1_ix)
            if extra_loc2_ix != 0:
                order += ix_to_loc(extra_loc2_ix)
            if extra_loc3_ix != 0:
                order += ix_to_loc(extra_loc3_ix)
            if extra_loc4_ix != 0:
                order += ix_to_loc(extra_loc4_ix)

        return ' '.join(order)


def ix_to_order(order_ix: int) -> str:
    return ACTION_LIST[order_ix]


def order_to_ix(order: str) -> int:
    if 'VIA' in order:
        order = order.split('VIA')[0]
        order += 'VIA'
    return ACTION_LIST.index(order)


def bits_between(number: int, start: int, end: int):
    """Returns bits between positions start and end from number."""
    return number % (1 << end) // (1 << start)
