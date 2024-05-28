import chat_config
from chat_config import ChatConfig

import pyttsx3
import os
import multiprocessing  # Para mejor manejo del Text2Speech

config = ChatConfig()
current_speak_process = None


# Preparando para la salida en pantalla
def send_chat_to_gui(modo, output_LLM, prompt, interfaz):
    if modo == 'manual' or modo == 'intro':
        # Se añade el propio prompt del usuario al Chat
        interfaz.ui.textBrowserOutput.append(config.name_user + ": " + prompt + "\n")

    # Sea intro, manual o automático, muestra la respuesta del LLM en el textBrowserOutput
    interfaz.ui.textBrowserOutput.append(config.name_bot + ": " + output_LLM + "\n")

    if config.text2speech_index != 0:  # lector de voz activado
        speak(output_LLM)


def spanish_reader(text, voice_id):
    engine = pyttsx3.init()
    if voice_id:
        engine.setProperty('voice', voice_id)
        engine.setProperty('rate', 109)
    engine.say(text)
    engine.runAndWait()


def english_reader(text, voice_id):
    engine = pyttsx3.init()
    if voice_id:
        engine.setProperty('voice', voice_id)
        engine.setProperty('rate', 99)
    engine.say(text)
    engine.runAndWait()


def speak(text):  # usaremos de lector la libreria de pyttsx3 - que a su vez usa voces instaladas en el SO.
    global current_speak_process

    # Si hay un proceso en ejecución, lo detenemos. Va bien si el modelo quiere hablar mucho en poco tiempo,
    # imitando un poco la ilusión de una persona a la que se le acumulan las ideas.
    if current_speak_process and current_speak_process.is_alive():
        current_speak_process.terminate()
        current_speak_process.join()

    if config.text2speech_index == 1:  # pyttsx3 con voz EN-US
        voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
        current_speak_process = multiprocessing.Process(target=spanish_reader, args=(text, voice_id))
    elif config.text2speech_index == 2:  # pyttsx3 con voz ES-ES
        voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-ES_HELENA_11.0"
        current_speak_process = multiprocessing.Process(target=english_reader, args=(text, voice_id))

    # Iniciar el proceso de habla
    current_speak_process.start()