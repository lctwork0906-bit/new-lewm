from enum import Enum
from typing import Dict, Union
import attr
import argparse


class Singleton(type):
    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]

class _DefaultAirsimActionSettings(Dict):
    FORWARD_STEP_SIZE = 2.0
    UP_DOWN_STEP_SIZE = 1.0
    LEFT_RIGHT_STEP_SIZE = 2.0
    TURN_ANGLE = 15.0
   
AirsimActionSettings: _DefaultAirsimActionSettings = _DefaultAirsimActionSettings()
