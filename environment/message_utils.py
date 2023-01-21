import torch

from environment.constants import POWER_ACRONYMS
from environment.message_list import MESSAGE_LIST
from environment.order_utils import order_to_daide_format


def send_message(sender, recipient, message):
    pass


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


def ix_to_msg(ix, game, power_name, last_message=None):
    """
    Gets a message from MESSAGE_LIST based on its index and formats it properly

    :param ix: index of the message
    :param game: game object
    :param power_name: name of the power sending the message
    :param last_message: last message, only applicable if the message is a reply
    :return: formatted message
    """
    msg = MESSAGE_LIST[ix]

    if msg[0] in ["YES", "REJ"]:
        return f"{msg[0]} ( {last_message} )"
    elif msg[0] == "PCE":
        return f"PRP ( PCE ( {power_list_to_acronyms(msg[1])} ) )"
    elif msg[0] == "ALY":
        return f"PRP (ALY ({power_list_to_acronyms(msg[1])}) VSS ({power_list_to_acronyms(msg[2])}))"
    elif msg[0] == "XDO":
        return f"PRP (XDO ({order_to_daide_format(msg[1], game, power_name)}))"
    elif msg[0] == "DMZ":
        return f"PRP (DMZ ({msg[1]}))"


def power_list_to_acronyms(power_list):
    """
    Converts a list of power names to their corresponding acronyms

    :param power_list: list of power names
    :return: string of power acronyms separated by spaces
    """
    return ' '.join([POWER_ACRONYMS[power] for power in power_list])


def filter_messages(dist, game, power_name):
    """
    Filters the message probabilities according to the current game state and power name

    :param dist: probability distribution of sending each message
    :param game: game object
    :param power_name: name of the power sending the message
    :return: filtered probability distribution of messages
    """
    msg_mask = torch.ones(len(MESSAGE_LIST))
    units = game.get_units(power_name)

    for i, msg in enumerate(MESSAGE_LIST):
        if msg[0] in ['PCE', 'ALY'] and power_name in msg[1]:
            msg_mask[i] = 1
        elif msg[0] == 'XDO' and ' '.join(msg[1].split()[:2]) in units:
            msg_mask[i] = 1

    dist = dist.masked_fill(msg_mask, value=0)

    return dist
