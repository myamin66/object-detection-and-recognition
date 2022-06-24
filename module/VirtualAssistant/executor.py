# Purpose: Execute user's command
from inputOutput import takeCommand
from intentValidator import *
from responseConstants import *
from utility import *
from inputOutput import speak
import os


def executeGreeting(engine, command=None):
    speak(engine, GREETING)
    return Modes.Waiting


def executeByeCommand(engine, command=None):
    speak(engine, "Goodbye for now! To wake me up, just say Hey")
    return Modes.Iddle


def executeShowInstructionCommand(engine, command):
    speak(engine, SELECT_MODE)
    return Modes.Waiting


def handleNearestObjectDirection(engine, command):
    os.system("python c:/Users/nomie/Desktop/Aura/ModeSelection/intel.py 1") 
    return Modes.Waiting

def handleFindObject(engine, command):
    speak(engine, 'Please say Where is my, followed by the object name')

    while True:
        query = takeCommand().lower()
        if isByeCommand(query):
            executeByeCommand(engine)
            return Modes.Iddle
        elif isShowInstructionCommand(query):
            executeShowInstructionCommand(engine)
        elif isWhereIsCommand(query):
            objName = extractObjectFromWhereIsCommand(query)
            if objName:
                speak(engine, 'You want to look for ' + objName)
                speak(engine, 'Please turn around slightly for to find the object')
                os.system("python c:/Users/nomie/Desktop/Aura/ModeSelection/intel.py 2 {name}".format(name=objName)) 
            return Modes.Waiting
        else:
            executeUnknownCommand(engine)


def handleDescribeView(engine, command):
    speak(engine, 'Please say capture')
    while True:
        query = takeCommand().lower()
        if isByeCommand(query):
            executeByeCommand(engine)
            return Modes.Iddle
        elif isShowInstructionCommand(query):
            executeShowInstructionCommand(engine)
        elif isWhatIsCommand(query):
            os.system("python c:/Users/nomie/Desktop/Aura/ModeSelection/intel.py 3") 
            return Modes.Waiting
        else:
            executeUnknownCommand(engine)


def executeUnknownCommand(engine, command=None):
    speak(engine, UNKNOWN)


def executeSelectModeCommand(engine, command):
    mode = extractModeFromCommand(command)
    if not isValidMode(mode):
        speak(engine, INVALID_MODE)
        return

    speak(engine, 'You select mode ' + mode)

    if MODES_MAP[mode] == 1:
        speak(engine, 'In this mode, I will help you avoid objects')
        return handleNearestObjectDirection(engine, command)
    elif MODES_MAP[mode] == 2:
        speak(engine, 'In this mode, I will help you find the object you want.')
        return handleFindObject(engine, command)
    else:
        speak(engine, 'I will describe what is in front of you')
        return handleDescribeView(engine, command)
