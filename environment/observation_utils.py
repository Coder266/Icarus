import numpy as np

from diplomacy import Game
from environment.action_list import ACTION_LIST
from environment.constants import *
from environment import order_utils

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


def get_season_code(state: dict) -> int:
    """
    Returns the season code for a given state name

    :param state_name: name of the state
    :return: season code
    :raises: ValueError if the state name is not recognized
    """
    state_name = state['name']
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


def get_loc_types():
    """
    Returns a dictionary with keys as location and values as the type of location (land, sea, coast)
    :return: dictionary with location and its corresponding type
    """
    return {key.upper(): item for key, item in Game().map.loc_type.items()}


def get_owner_by_loc(state: object):
    """
    Given a state, returns a dictionary with keys as location and values as the power that controls that location
    :param state: current state of the game
    :return: dictionary with location and its corresponding power
    """
    return {loc: power for power in state["influence"] for loc in state["influence"][power]}


def get_unit_type_by_loc(state: dict):
    """
    Given a state, returns a dictionary with keys as location and values as the unit type present in that location
    :param state: current state of the game
    :return: dictionary with location and its corresponding unit type
    """
    units = sum(state["units"].values(), [])
    units = [unit.split(' ') for unit in units]
    units = {unit[1]: unit[0] for unit in units}

    return units


def get_unit_type_in_loc(loc: str, state: dict) -> str:
    """
    Given a location and a state, returns the unit type present in that location
    :param loc: location for which unit type is to be returned
    :param state: current state of the game
    :return: unit type present in the location
    """
    units = get_unit_type_by_loc(state)

    if loc in units:
        unit_type = units[loc]
    else:
        unit_type = ' '
        # raise ValueError(f"No unit in location {loc}")

    return unit_type


def get_centers_by_loc(state):
    """
    Given a state, returns a dictionary with keys as supply center locations
    and values as the power that controls supply center
    :param state: current state of the game
    :return: dictionary with location and its corresponding power
    """
    return {center: power for power in state["centers"] for center in state["centers"][power]}


def get_dislodged_units_by_loc(state):
    """
    Given a state, returns a dictionary with keys as location and values as the unit type dislodged from that location
    :param state: current state of the game
    :return: dictionary with location and its corresponding unit type
    """
    return {retreats.split()[1]: retreats.split()[0]
            for power in state["retreats"]
            for retreats in state["retreats"][power]}


def get_dislodged_units_power_by_loc(state):
    """
    Given a state, returns a dictionary with keys as location
    and values as the power that the dislodged unit on that location belongs to
    :param state: current state of the game
    :return: dictionary with location and its corresponding power
    """
    return {retreats.split()[1]: power for power in state["retreats"] for retreats in state["retreats"][power]}


def get_board_state(state):
    """
    Given the state of a game, this function returns an array representing the state of the board.
    The array has dimensions (num_locs, loc_vector_length) and contains information about the unit type,
    power influence, buildable status, removable status, dislodged units, location type, and
    whether it is a supply center for each location on the board.
    Each location vector is composed of zeros

    :param state: current state of the game.
    :return: an array of shape (num_locs, loc_vector_length) containing information about the state of the board.
    """

    # Initialize empty board state
    board_state = []

    # Get information about the unit types, power influence, dislodged units, supply centers, and location types
    unit_type = get_unit_type_by_loc(state)
    owner = get_owner_by_loc(state)
    dislodged_powers = get_dislodged_units_power_by_loc(state)
    dislodged_units = get_dislodged_units_by_loc(state)
    centers = get_centers_by_loc(state)
    loc_types = get_loc_types()

    # Iterate over each location on the board
    for loc in LOCATIONS:
        # Initialize empty location vector
        loc_vector = np.zeros(LOC_VECTOR_LENGTH)

        # Unit type
        if loc in unit_type:
            if unit_type[loc] == 'A':
                loc_vector[UNIT_TYPE_INDEX] = 1
            elif unit_type[loc] == 'F':
                loc_vector[UNIT_TYPE_INDEX + 1] = 1
        else:
            loc_vector[UNIT_TYPE_INDEX + 2] = 1

        # Power
        if loc in owner:
            loc_vector[POWER_INDEX + ALL_POWERS.index(owner[loc])] = 1
        else:
            loc_vector[POWER_INDEX + 7] = 1

        # Buildable
        if loc in owner:
            loc_vector[BUILDABLE_INDEX] = int(loc in state["builds"][owner[loc]]["homes"])

        # Removable
        if loc in owner:
            loc_vector[REMOVABLE_INDEX] = any(loc in unit for unit in state["units"][owner[loc]])

        # Dislodged
        if loc in dislodged_units:
            if dislodged_units[loc] == 'A':
                loc_vector[DISLODGED_UNIT_INDEX] = 1
            elif dislodged_units[loc] == 'F':
                loc_vector[DISLODGED_UNIT_INDEX + 1] = 1
            loc_vector[DISLODGED_POWER_INDEX + ALL_POWERS.index(dislodged_powers[loc])] = 1
        else:
            loc_vector[DISLODGED_UNIT_INDEX + 2] = 1
            loc_vector[DISLODGED_POWER_INDEX + 7] = 1

            # Location type
            loc_vector[LAND_TYPE_INDEX + LAND_TYPES.index(loc_types[loc])] = 1

            # Center owner
            if loc in centers:
                loc_vector[CENTER_OWNER_INDEX + ALL_POWERS.index(centers[loc])] = 1
            else:
                loc_vector[CENTER_OWNER_INDEX + 7] = 1

            # Append location vector to board state
            board_state.append(loc_vector)

        # Return board state as numpy array
        return np.array(board_state)


def get_last_phase_orders(game: Game):
    """
    Given a game object, this function returns an array representing the orders of the last phase.
    The array has dimensions (num_locs, order_size)
    and contains information about the order issued on each location on the board.

    :param game: game object
    :return: an array of shape (num_locs, order_size) containing information about the orders of the last phase.
    """
    phase_history = Game.get_phase_history(game, from_phase=-1)
    if phase_history:
        return phase_orders_to_rep(phase_history[0].orders)
    else:
        return phase_orders_to_rep([])


def phase_orders_to_rep(phase_orders):
    """
    Given the list of orders of a phase, this function returns an array representing the orders.
    The array has dimensions (num_locs, order_size) and contains information
    about the order type, unit type, and target location for each location on the board.

    :param phase_orders: a list of strings representing the orders of a phase.
    :return: an array of shape (num_locs, order_size) containing information about the orders of the phase.
    """
    if not phase_orders:
        return np.zeros((len(LOCATIONS), order_utils.ORDER_SIZE))

    phase_orders = sum(phase_orders.values(), [])
    order_by_loc = {order.split()[1]: order_utils.order_to_rep(order) for order in phase_orders
                    if order != 'WAIVE' and order in ACTION_LIST}

    phase_orders_rep = [order_by_loc[loc] if loc in order_by_loc else 0 for i, loc in enumerate(LOCATIONS)]

    return phase_orders_rep
