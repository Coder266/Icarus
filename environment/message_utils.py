from environment.action_list import ACTION_LIST
from environment.constants import POWER_ACRONYMS
from environment.message_list import MESSAGE_LIST
from environment.order_utils import order_to_daide_format


def send_message(sender, recepient, message):
    pass


def msg_to_emb():
    pass


def ix_to_msg(ix, game, power_name, last_message=None):
    message_rep = MESSAGE_LIST[ix]

    if message_rep[0] in ["YES", "REJ"]:
        return f"{message_rep[0]} ({last_message})"
    elif message_rep[0] == "PCE":
        return f"PRP (PCE ({' '.join(message_rep[1])}))"
    elif message_rep[0] == "ALY":
        return f"PRP (ALY ({' '.join(message_rep[1])}) VSS ({' '.join(message_rep[2])}))"
    elif message_rep[0] == "XDO":
        return f"PRP (XDO ({order_to_daide_format(message_rep[1], game, power_name)}))"
    elif message_rep[0] == "DMZ":
        return f"PRP (DMZ ({message_rep[1]}))"
    else:
        raise Exception


def filter_messages(dist, power_name, game):
    # orderable_locs = game.get_orderable_locations()
    #
    # # filter invalid orders
    # dist_clone = dist.clone().detach()
    # for i, loc in enumerate(orderable_locs[power_name]):
    #     order_mask = torch.ones_like(dist_clone[i], dtype=torch.bool)
    #     order_mask[get_loc_valid_orders(game, loc)] = False
    #     dist_clone[i, :] = dist_clone[i, :].masked_fill(order_mask, value=0)
    #
    # dist_clone = Categorical(dist_clone)
    #
    # actions = dist_clone.sample()
    # return actions
    pass
