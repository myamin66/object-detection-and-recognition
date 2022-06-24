# Purpose: Find the user's intention
from utility import VALID_MODES

def isGreetingCommand(command):
    greetingWords = ['hello', 'hi']
    if any([word in command for word in greetingWords]):
        return True
    return False


def isByeCommand(command):
    byeWords = ['bye', 'goodbye']
    if any([word in command for word in byeWords]):
        return True
    return False

def isWhereIsCommand(command):
    return 'where is' in command or "where's" in command

def isWhatIsCommand(command):
    return 'capture' in command or "capture" in command

def isSelectModeCommand(command):
    return any([mode in command for mode in VALID_MODES])


def isShowInstructionCommand(command):
    return 'instruction' in command


def isWakeUpCommand(command):
    # TODO: hey Aura is difficult to pronounce lol
    # Putting hey Google here for testing purpose
    # Should change to hey Aura later
    return 'hey' in command or 'hey' in command
