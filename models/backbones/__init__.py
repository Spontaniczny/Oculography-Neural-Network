from .res_net import create_res_net_18, create_res_net_34, create_res_net_50 
from .backbone import Backbone


def init_backbone(backbone: str) -> Backbone:
    match backbone:
        case "res_net_18":
            return create_res_net_18()
        case "res_net_34":
            return create_res_net_34()
        case _:
            return create_res_net_50()
        