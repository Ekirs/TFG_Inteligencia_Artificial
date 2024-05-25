import chat_config
from chat_config import ChatConfig

chat_config_ = ChatConfig()


# preparando para la salida en pantalla
def send_chat_to_gui(modo, output_LLM, prompt, interfaz):
    if modo == 'manual' or modo == 'intro':
        # se añade el propio prompt del usuario al Chat
        interfaz.ui.textBrowserOutput.append(chat_config_.name_user + ": " + prompt + "\n")

    # sea intro, manual o automático, muestra la respuesta del LLM en el textBrowserOutput
    interfaz.ui.textBrowserOutput.append(chat_config_.name_bot + ": " + output_LLM + "\n")