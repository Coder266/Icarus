from environment.action_list import ACTION_LIST
from environment.constants import ALL_POWERS, LOCATIONS
from environment.message_utils import action_list_to_XDO_format

PRESS_LEVEL = 1

MESSAGE_TOKENS = []

if PRESS_LEVEL >= 1:
    MESSAGE_TOKENS += [
        "ANS",  # used when responding, copies last received message
        "YES",
        "REJ",
        "BWX",
        "CCL",
        "PRP",
        "NOT",
        "NAR",
        "DRW",
        "PCE",
        "ALY",
        "VSS",
        *ALL_POWERS,
        "FCT"
    ]

if PRESS_LEVEL >= 2:
    MESSAGE_TOKENS += [
        "XDO",
        action_list_to_XDO_format(),
        "DMZ",
        *LOCATIONS
    ]
