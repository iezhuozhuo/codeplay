# -*- coding: utf-8 -*-
# @Author: zhuo & zdy
# @github: iezhuozhuo
# @vaws: Making Code Great Again!

# from source.models.xxx import *


def ModelConfig(parser):
    args, _ = parser.parse_known_args()
    return args


def XXXModel(args, embedd):
    model = xxx()
    return model