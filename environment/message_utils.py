import re

import torch

from environment.constants import POWERS_TO_ACRONYMS, ACRONYMS_TO_POWERS, ALL_POWERS
from environment.message_list import MESSAGE_LIST, ANSWER_LIST
from environment.order_utils import order_to_daide_format, get_unit_owner, daide_order_to_order_format


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
        return f"PRP ( DMZ ( {msg[1]} ) )"


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
        msg = ['DMZ', tokens[4]]
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


def filter_messages(dist, game, power_name):
    """
    Filters the message probabilities according to the current game state and power name

    :param dist: probability distribution of sending each message
    :param game: game object
    :param power_name: name of the power sending the message
    :return: filtered probability distribution of messages
    """
    msg_mask = torch.ones(len(MESSAGE_LIST), dtype=torch.bool).to(dist.device)
    units = game.get_units(power_name)

    for i, msg in enumerate(MESSAGE_LIST):
        if msg[0] in ['PCE', 'ALY'] and power_name in msg[1]:
            msg_mask[i] = True
        elif msg[0] == 'XDO' and ' '.join(msg[1].split()[:2]) in units:
            msg_mask[i] = True

    dist = dist.masked_fill(msg_mask, value=0)

    return dist


def get_message_targets(game, msg):
    if msg[0] in ['PCE', 'ALY']:
        return [POWERS_TO_ACRONYMS[power] for power in msg[1]]
    elif msg[0] == 'XDO':
        return get_unit_owner(game, ' '.join(msg[1].split()[:2]))
    else:
        raise ValueError(f'Unable to determine target in message {msg}')


async def send_message(game, power_name, msg_ix, last_message=None, reply_power=None):
    if reply_power:
        targets = [reply_power]
    else:
        msg = ix_to_msg(msg_ix)
        targets = get_message_targets(game, msg)

    daide_msg = ix_to_daide_msg(msg_ix, game, power_name, last_message)
    for target in targets:
        msg_object = game.new_power_message(target, daide_msg)
        await game.send_game_message(message=msg_object)

        print(f'Sent message {daide_msg}')
