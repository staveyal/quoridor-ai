"""
This code was downloaded from https://www.pypi.org/project/python-quoridor
And we changed it to match our needs
All right reserved to the original writer due to MIT license
"""
import string
from enum import Enum
from typing import List,Pattern
import re

ALL_QUORIDOR_MOVES_REGEX: Pattern = re.compile(r"[a-i][1-9](?:[hv])?")
START_POS_P1: str = "e1"
GOAL_P1: str = "9"
START_POS_P2: str = "e9"
GOAL_P2: str = "1"
START_WALLS: int = 10
POSSIBLE_WALLS: List[str] = [
    string.ascii_letters[i] + str(j) + c
    for i in range(8)
    for j in range(1, 9)
    for c in ["h", "v"]
]


class GameStatus(Enum):
    """
    Represents the possible statuses of a Quoridor game.
    """

    COMPLETED: str = "Completed"
    CANCELLED: str = "Canceled"
    ONGOING: str = "Ongoing"