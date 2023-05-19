import copy

import re

import torch

from environment.constants import POWERS_TO_ACRONYMS, ACRONYMS_TO_POWERS, ALL_POWERS, POWER_ACRONYMS_LIST, LOCATIONS, \
    DAIDE_LOCATIONS
from environment.message_list import MESSAGE_LIST, ANSWER_LIST
from environment.order_utils import order_to_daide_format, get_unit_owner, daide_order_to_order_format, \
    daide_format_to_loc


def msg_to_ix(msg):
    """
    Converts a message to its index in MESSAGE_LIST

    :param msg: message to be converted
    :return: index of the message in MESSAGE_LIST
    :raises: ValueError if the message is not in MESSAGE_LIST
    """
    if msg in MESSAGE_LIST:
        return MESSAGE_LIST.index(msg)
    else:
        raise ValueError(f"Message {msg} not in MESSAGE_LIST")


def ix_to_msg(ix):
    return MESSAGE_LIST[ix]


def ix_to_daide_msg(msg_ix, game, power_name, last_message=None):
    """
    Gets a message from MESSAGE_LIST based on its index and formats it properly

    :param msg_ix: index of the message
    :param game: game object
    :param power_name: name of the power sending the message
    :param last_message: last message, only applicable if the message is a reply
    :return: formatted message
    """

    if last_message:
        return f"{ANSWER_LIST[msg_ix]} ( {last_message} )"

    msg = MESSAGE_LIST[msg_ix]
    if msg[0] == "PCE":
        return f"PRP ( PCE ( {power_list_to_acronyms(msg[1])} ) )"
    elif msg[0] == "ALY":
        return f"PRP ( ALY ( {power_list_to_acronyms(msg[1])} ) VSS ( {power_list_to_acronyms(msg[2])} ) )"
    elif msg[0] == "XDO":
        return f"PRP ( XDO ( {order_to_daide_format(msg[1], game, power_name)} ) )"
    elif msg[0] == "DMZ":
        return f"PRP ( DMZ ( {' '.join([POWERS_TO_ACRONYMS[power] for power in msg[2]])} ) ( {msg[1]} ) )"


def get_daide_msg_ix(daide_msg):
    tokens = daide_msg.replace('(', ' ').replace(')', ' ').split()

    if tokens[1] == "PCE":
        msg = ['PCE', tuple(sorted([ACRONYMS_TO_POWERS[acro] for acro in tokens[2:]]))]
    elif tokens[1] == "ALY":
        vss_ix = tokens.index('VSS')
        msg = ['ALY', tuple(sorted([ACRONYMS_TO_POWERS[acro] for acro in tokens[2:vss_ix]])),
               tuple(sorted([ACRONYMS_TO_POWERS[acro] for acro in tokens[vss_ix+1:]]))]
    elif tokens[1] == "XDO":
        msg = ['XDO', daide_order_to_order_format(' '.join(tokens[2:]))]
    elif tokens[1] == "DMZ":
        powers = []
        loc = None
        for token in tokens:
            if token in POWER_ACRONYMS_LIST:
                powers.append(ACRONYMS_TO_POWERS[token])
            elif token in DAIDE_LOCATIONS:
                loc = daide_format_to_loc(token)

        if powers and loc:
            msg = ['DMZ', loc, tuple(sorted(powers))]
        else:
            raise ValueError(f"Invalid DMZ message {daide_msg}")
    else:
        raise ValueError(f"Unknown daide message {daide_msg}")

    return msg_to_ix(msg)


def is_daide_msg_reply(last_message):
    # alternatively split into tokens and check first token
    return 'YES' in last_message or 'REJ' in last_message or 'BWX' in last_message


def get_msg_ixs_from_daide_reply(last_message):
    tokens = last_message.replace('(', ' ').replace(')', ' ').replace('BWX', 'REJ').split()
    if tokens[0] in ANSWER_LIST:
        msg_ix = ANSWER_LIST.index(tokens[0])
        answered_msg_ix = get_daide_msg_ix(' '.join(tokens[1:]))

        return msg_ix, answered_msg_ix
    else:
        return None, None


def power_list_to_acronyms(power_list):
    """
    Converts a list of power names to their corresponding acronyms

    :param power_list: list of power names
    :return: string of power acronyms separated by spaces
    """
    return ' '.join([POWERS_TO_ACRONYMS[power] for power in power_list])


def filter_messages(dist, power_name, units):
    """
    Filters the message probabilities according to the current game state and power name

    :param dist: probability distribution of sending each message
    :param power_name: name of the power sending the message
    :param units: units controlled by power
    :return: filtered probability distribution of messages
    """
    msg_mask = torch.ones(len(MESSAGE_LIST), dtype=torch.bool).to(dist.device)

    for i, msg in enumerate(MESSAGE_LIST):
        if msg[0] in ['PCE', 'ALY'] and power_name in msg[1]:
            msg_mask[i] = False
        elif msg[0] == 'XDO' and ' '.join(msg[1].split()[:2]) in units:
            msg_mask[i] = False
        elif msg[0] == 'DMZ' and len(msg[2]) == 2 and power_name not in msg[2]:
            msg_mask[i] = False

    dist = dist.masked_fill(msg_mask, value=0)

    return dist


def get_message_targets(game, power_name, msg):
    if msg[0] in ['PCE', 'ALY']:
        return [POWERS_TO_ACRONYMS[power] for power in msg[1] if power != power_name]
    elif msg[0] == 'XDO':
        return get_unit_owner(game, ' '.join(msg[1].split()[:2]))
    elif msg[0] == 'DMZ':
        return [POWERS_TO_ACRONYMS[power] for power in msg[2] if power != power_name]
    else:
        raise ValueError(f'Unable to determine target in message {msg}')


async def send_message(game, power_name, msg_ix, last_message=None, reply_power=None, press_allowed_powers=ALL_POWERS):
    if reply_power:
        targets = [reply_power]
    else:
        msg = ix_to_msg(msg_ix)
        targets = get_message_targets(game, power_name, msg)

    daide_msg = ix_to_daide_msg(msg_ix, game, power_name, last_message)
    for target in targets:
        if target in POWER_ACRONYMS_LIST:
            target = ACRONYMS_TO_POWERS[target]

        if target in press_allowed_powers:
            msg_object = game.new_power_message(target, daide_msg)

            await game.send_game_message(message=msg_object)

            print(f'Sent message {daide_msg}')


def split_Albert_DMZs(received_messages):
    split_messages = []
    for i, msg_obj in enumerate(received_messages):
        new_msg_texts = split_DMZ(msg_obj['message'])
        for msg_text in new_msg_texts:
            new_msg_obj = copy.deepcopy(msg_obj)
            new_msg_obj['message'] = msg_text
            split_messages.append(new_msg_obj)
    return split_messages


def split_DMZ(message):
    messages = []
    tokens = message.replace('(', ' ').replace(')', ' ').split()
    if 'DMZ' in tokens:
        powers = []
        locs = []
        for token in tokens:
            if token in POWER_ACRONYMS_LIST:
                powers.append(token)
            elif token in DAIDE_LOCATIONS:
                locs.append(token)

        if len(locs) >= 2:
            for loc in locs:
                if tokens[1] == 'DMZ':
                    new_message = \
                        f"PRP ( DMZ ( {' '.join(powers)} ) ( {loc} ) )"
                elif tokens[2] == 'DMZ':
                    new_message = \
                        f"{tokens[0]} ( PRP ( DMZ ( {' '.join(powers)} ) ( {loc} ) ) )"
                else:
                    raise ValueError(f"Unknown DMZ message: {message}")

                messages.append(new_message)

    if not messages:
        messages = [message]

    return messages