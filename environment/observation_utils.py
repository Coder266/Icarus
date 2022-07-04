import numpy as np
from diplomacy import Game
from environment.constants import *
import environment.order_utils as order_utils

# max num orders * num powers
MAX_ORDERS = 18
NUM_ORDERS_PER_PHASE = MAX_ORDERS * len(ALL_POWERS)
LOC_VECTOR_LENGTH = 35

UNIT_TYPE_INDEX = 0
POWER_INDEX = 3
BUILDABLE_INDEX = 11
REMOVABLE_INDEX = 12
DISLODGED_UNIT_INDEX = 13
DISLODGED_POWER_INDEX = 16
LAND_TYPE_INDEX = 24
CENTER_OWNER_INDEX = 27
CENTER_INDEX = 34


def get_season_code(state_name: str) -> int:
    if state_name[0] == 'S' and state_name[-1] == 'M':
        return SPRING_MOVES
    elif state_name[0] == 'S' and state_name[-1] == 'R':
        return SPRING_RETREATS
    if state_name[0] == 'F' and state_name[-1] == 'M':
        return FALL_MOVES
    elif state_name[0] == 'F' and state_name[-1] == 'R':
        return FALL_RETREATS
    elif state_name[0] == 'W':
        return WINTER_ADJUSTMENTS
    elif state_name == 'COMPLETED':
        return COMPLETED
    else:
        raise ValueError(f"Unknown season in state {state_name}")


def get_owner_by_loc(state):
    return {loc: power for power in state["influence"] for loc in state["influence"][power]}


def get_unit_type_by_loc(game):
    units = sum(game.get_state()["units"].values(), [])
    units = [unit.split(' ') for unit in units]
    units = {unit[1]: unit[0] for unit in units}

    return units


def get_unit_type_in_loc(loc: str, game: Game) -> str:
    units = get_unit_type_by_loc(game)

    if loc in units:
        unit_type = units[loc]
    else:
        unit_type = ' '
        # raise ValueError(f"No unit in location {loc}")

    return unit_type


def get_centers_by_loc(state):
    return {center: power for power in state["centers"] for center in state["centers"][power]}


def get_dislodged_units_by_loc(state):
    return {retreats.split()[1]: retreats.split()[0]
            for power in state["retreats"]
            for retreats in state["retreats"][power]}


def get_dislodged_units_power_by_loc(state):
    return {retreats.split()[1]: power for power in state["retreats"] for retreats in state["retreats"][power]}


def get_loc_types(game):
    return {key.upper(): item for key, item in game.map.loc_type.items()}


def get_observation(game: Game):
    state = game.get_state()

    # season
    season = get_season_code(state["name"])

    # board
    # Array num_areas x loc_vector_length
    board_state = []
    unit_type = get_unit_type_by_loc(game)
    owner = get_owner_by_loc(state)
    dislodged_powers = get_dislodged_units_power_by_loc(state)
    dislodged_units = get_dislodged_units_by_loc(state)
    centers = get_centers_by_loc(state)
    loc_types = get_loc_types(game)

    for loc in LOCATIONS:
        # UNIT_TYPE (3 flags) (army, fleet, none) | POWER (8 flags) (power + none) | BUILDABLE (1 flag) |
        # REMOVABLE (1 flag) | DISLODGED (3 flags) (army, fleet, none) | OWNER OF DISLODGED (8 flags) (power + none) |
        # LOC TYPE (3 flags) (land, sea, coast) | OWNER OF SUPPLY CENTER (8 flags) (power + none)

        # loc_vector
        loc_vector = np.zeros(LOC_VECTOR_LENGTH)

        # unit type
        if loc in unit_type:
            loc_vector[UNIT_TYPE_INDEX] = unit_type[loc] == 'A'
            loc_vector[UNIT_TYPE_INDEX + 1] = unit_type[loc] == 'F'
        else:
            loc_vector[UNIT_TYPE_INDEX + 2] = True

        # power
        if loc in owner:
            loc_vector[POWER_INDEX + ALL_POWERS.index(owner[loc])] = True
        else:
            loc_vector[POWER_INDEX + 7] = True

        # buildable
        if loc in owner:
            loc_vector[BUILDABLE_INDEX] = loc in state["builds"][owner[loc]]["homes"]

        # removable
        if loc in owner:
            loc_vector[REMOVABLE_INDEX] = any(loc in unit for unit in state["units"][owner[loc]])

        # dislodged
        if loc in dislodged_units:
            # dislodged[loc]
            loc_vector[DISLODGED_UNIT_INDEX] = dislodged_units[loc] == 'A'
            loc_vector[DISLODGED_UNIT_INDEX + 1] = dislodged_units[loc] == 'F'

            loc_vector[DISLODGED_POWER_INDEX + ALL_POWERS.index(dislodged_powers[loc])] = True
        else:
            loc_vector[DISLODGED_UNIT_INDEX + 2] = True
            loc_vector[DISLODGED_POWER_INDEX + 7] = True

        # loc type
        loc_vector[LAND_TYPE_INDEX] = LAND_TYPES.index(loc_types[loc])

        # centers
        if loc in centers:
            loc_vector[CENTER_OWNER_INDEX + ALL_POWERS.index(centers[loc])] = True
        else:
            loc_vector[CENTER_OWNER_INDEX + 7] = True

        board_state += [loc_vector]

    board_state = np.concatenate(board_state)

    # build_number
    # vector of length 7 with the number of units a player can build/has to remove
    build_numbers = [builds["count"] for builds in state["builds"].values()]

    # last phase orders
    if game.get_phase_history(-1):
        last_phase_orders = [order_utils.order_to_id(order) for order in sum(game.get_phase_history(-1)[0].orders.values(), [])]
    else:
        last_phase_orders = []
    last_phase_orders = np.pad(last_phase_orders, [0, NUM_ORDERS_PER_PHASE - len(last_phase_orders)])

    return np.concatenate([[season], board_state, build_numbers, last_phase_orders])


def get_observation_length():
    # season, board_state, build_numbers, last_actions
    return 1 + (len(LOCATIONS) * LOC_VECTOR_LENGTH) + len(ALL_POWERS) + NUM_ORDERS_PER_PHASE
