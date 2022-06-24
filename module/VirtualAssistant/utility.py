import enum


class Modes(enum.Enum):
    Iddle = 0
    Waiting = 4


VALID_MODES = set(['one', 'two', 'three', '1', '2', '3'])
MODES_MAP = {'one': 1, '1': 1, 'two': 2, '2': 2, 'three': 3, '3': 3}


def extractModeFromCommand(command):
    words = command.split()
    if len(words) == 0:
        return "None"
    return words[0]


def extractObjectFromWhereIsCommand(command):
    words = command.split()
    for i in range(len(words) - 2):
        if i < len(words) - 3 and words[i] == 'where' and words[i + 1] == 'is':
            return words[i + 3]
        elif words[i] == "where's":
            return words[i + 2]
    return None


def isValidMode(mode):
    return mode in VALID_MODES
