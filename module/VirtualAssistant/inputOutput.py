import pyttsx3
import speech_recognition as sr

PAUSE_THRESHOLD = 0.7
DEFAULT_LANGUAGE_RECOGNITION = 'en-US'
VOICE_GENDER = {'male': 0, 'female': 1}


def takeCommand():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print('Listening...')

        r.pause_threshold = PAUSE_THRESHOLD
        audio = r.listen(source)

        try:
            print('Recognizing')
            query = r.recognize_google(
                audio, language=DEFAULT_LANGUAGE_RECOGNITION)
            print('Query:', query)
        except Exception as e:
            print(e)
            return "None"
        return query


def initSpeakEngine(gender):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[VOICE_GENDER[gender]].id)
    return engine


def speak(engine, audio):
    engine.say(audio)

    # Blocks while processing all the queued command
    engine.runAndWait()
