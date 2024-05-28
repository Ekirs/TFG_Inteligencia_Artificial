from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # modelos locales
import time, threading  # Chat automatizado
import re  # Trabajado de prompts

import cohere  # Modelo online usable en este proyecto

# IMPORTANTE - archivo único para cada usuario. Bastará crear un online_config.py con el string cohere_key.
from online_config import cohere_key  # key intransferible de cada usuario, para API Cohere

from utils import send_chat_to_gui
from chat_config import ChatConfig


class LLM_Modelo:
    def __init__(self):
        self.myModel = None
        self.myTokenizer = None
        self.onlineModel = None
        self.log_string = ""
        self.chat_log = []


config = ChatConfig()  # Instancia para variables generales

llm = LLM_Modelo()  # Instancia para interactuar con modelos

# Mensajes de ejemplo para modelos
llm.chat_log = [{"role": config.name_user, "message": "Tell me something about birds."},
                {"role": config.name_bot, "message": "Flamingos are birds, and a group of flamingos is called a "
                                                     + "flamboyance,"
                                                     + "fittingly reflects their vibrant pink color and unique "
                                                     + "appearance."},
                {"role": config.name_user, "message": "Tell me some food recipy."},
                {"role": config.name_bot, "message": "Here's a super quick and easy recipe: "
                                                     + "mash 1/2 avocado and spread on toast, "
                                                     + "sprinkle salt and pepper to taste, "
                                                     + "and add optionally a squeeze of lemon juice and a sprinkle of "
                                                       "red pepper flakes. Ready in 2 minutes!"},
                {"role": config.name_user, "message": "Tell me something about barcelona."},
                {"role": config.name_bot,
                 "message": "Did you know that Barcelona has a secret beach hidden beneath the city? The beach is "
                            + "called Nova Icària, and it's a small, secluded spot that's tucked away from the main "
                            + "tourist areas."},
                {"role": config.name_user, "message": "Tell me more about it."},
                {"role": config.name_bot,
                 "message": "Another beach, would be the Barceloneta Beach: This is one of the most popular and "
                            + "iconic beaches in Barcelona. It's"
                            + " a urban beach, meaning it's located in the heart of the city, and it's known for its "
                            + "lively atmosphere, restaurants, and bars."},
                {"role": config.name_user, "message": "Tell me more about it."},
                {"role": config.name_bot,
                 "message": "If you go to Barcelona, you should check Barcelona's nightlife! It's a city that truly "
                            + "comes alive after dark, with a vibrant and"
                            + " eclectic mix of bars, clubs, and live music venues."}
                ]

# Variable con el tiempo del modo auto. Accesible desde otras clases para interrumpirlo.
temp_timer = 0


def my_model():
    if config.llm_index_gui == 0:
        config.model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
        config.modelo_activo = "mistral-7b"
        config.modelo_licencia = "Licencia Apache 2.0"

    elif config.llm_index_gui == 1:
        config.model_name_or_path = "TheBloke/Llama-2-7b-Chat-GPTQ"
        config.modelo_activo = "llama2-7b"
        config.modelo_licencia = ("Built with Meta Llama 2. \nMeta Llama 2 is "
                                  + "licensed under the Meta Llama 2 Community License, "
                                  + "Copyright © Meta Platforms, Inc. All Rights Reserved.")
    elif config.llm_index_gui == 2:
        config.model_name_or_path = "Cohere Command. Modelo online."
        config.modelo_activo = "cohere"
        config.modelo_licencia = ("Uso puramente académico de Cohere Command para consultas."
                                  + "Mas información en https://cohere.com/legal/terms-of-service/ Copyright © Cohere Inc.")


def stop_model():
    # Liberamos memoria del modelo previo
    llm.myModel = None
    llm.myTokenizer = None
    llm.onlineModel = None
    config.stop_llm = False  # se libera la petición de pararlo, por si se busca iniciar otro


def auto_prompt(interfaz):
    print("Ejecutando método Auto Prompt.")
    if config.modelo_activo=="llama2-7b":
        config.prompt_auto = "Tell me some random trivia."
    else:
        config.prompt_auto = "Tell me more about that."

    send_prompt(config.prompt_auto)
    send_chat_to_gui("auto", config.output_LLM, config.prompt_auto, interfaz)


def start_timer(interfaz):  # preparando hilo para temporizador
    thread = threading.Thread(target=run_timer, args=(interfaz,))
    thread.start()


def run_timer(interfaz):  # temporizador en marcha
    print("\nTiempo temporizador es:" + str(config.timer_auto_llm) + "\n")
    # Almacena el valor inicial de config.timer_auto_llm
    temp_timer = config.timer_auto_llm  # config.timer_auto_llm, cambiable desde combobox de interfaz

    while temp_timer > 0 and not config.make_auto_llm_restart:
        time.sleep(5)  # Espera 5 segundos
        if config.stop_llm:
            return  # Sale de la función y para el modo auto, si stop_llm es True
        temp_timer -= 5

    print("\nReiniciado timer:" + str(config.make_auto_llm_restart) + "\n")
    if not config.make_auto_llm_restart:
        auto_prompt(interfaz)  # Llama a la función de devolución de llamada una vez que el temporizador haya terminado

    # Preparamos el reinicio del automatizado.
    config.make_auto_llm_restart = False  # Volverá a True en cuanto el usuario escriba algo
    if not config.stop_llm:
        start_timer(interfaz)  # Volvemos al temporizador, creando un bucle.


def build_prompt(prompt):
    config.prompt_en_proceso = True
    # Construimos prompt adecuado para un LLM. Le adjuntaremos el historial en un string.
    llm.log_string = ""
    for message in llm.chat_log:
        # IMPORTANTE - algunos modelos usarán el log de manera separada respecto al prompt.
        llm.log_string += f"{message['role']}: {message['message']}\n"

    # Le añadiremos el prompt a medida según el modelo elegido, para especializar el prompt.
    # opciones offline:
    if config.modelo_activo == "mistral-7b":
        return build_mistral_prompt(llm.log_string, prompt)

    elif config.modelo_activo == "llama2-7b":
        return build_ll2_prompt(llm.log_string, prompt)

    # Opciones online
    elif config.modelo_activo == "cohere":
        # Cohere no usa un prompt mezclado con el log, los trata por separado.
        return build_cohere_prompt(prompt)

    else:
        return prompt


def build_mistral_prompt(log_string, prompt):  # Mistral 7B v0.2 Instruct
    prompt_mistral = (
            "[INST]" + "You are a friendly entity called " + config.name_bot
            + ". The following is part of a conversation between you and " + config.name_user
            + ". Write a response that appropriately completes the request made after the string <<USER_PROMPT>>."
            + "You will try to answer briefly with less than 50 words, and you will never exceed the 100 words in "
            + "your answer. \n\n"
            + log_string + "\n<<USER_PROMPT>>:"
            + prompt + "[/INST]"
    )
    return prompt_mistral


def build_ll2_prompt(log_string, prompt):  # Llama 2 7B Chat
    prompt_ll2 = (
            "[INST]" + "<<SYS>>You are a friendly entity called " + config.name_bot
            + ". The following is part of a conversation between you and " + config.name_user
            + ". Write a response that appropriately completes the request made after the string <<USER_PROMPT>>."
            + "You will try to answer briefly with less than 50 words, and you will never exceed the 100 words in "
            + "your answer.<</SYS>> \n\n"
            + log_string + "\n\n<<USER_PROMPT>>:"
            + prompt + "[/INST]"
    )
    return prompt_ll2


def build_cohere_prompt(prompt):
    return prompt  # Cohere no tratará su prompt de usuario mezclándolo con instrucciones, o log.


def estimate_token_usage(chat_log):
    # Calcular la cantidad total de tokens en el registro
    token_estimate = sum(len(message["message"].split()) for message in chat_log)
    return token_estimate


def check_token_limit(chat_log, limit=config.context_size):
    # En caso que los tokens amenacen con rebosar, se efectua limpieza por el tramo superior
    while estimate_token_usage(chat_log) > (limit - 200):
        print(f"Estimación actual de tokens: {estimate_token_usage(chat_log)}")
        print(f"Límite de tokens: {limit}, cerca de rebosar")
        # Eliminar el primer mensaje del registro para hacer espacio
        chat_log.pop(0)
        print("Se eliminó el primer mensaje para hacer espacio.")

    print("Estimación de tokens dentro del límite.")


def respuesta_modelo(output_LLM):
    if config.modelo_activo == "mistral-7b":
        return respuesta_mistral(output_LLM)

    elif config.modelo_activo == "llama2-7b":
        # NOTA - por la similitud en prompt y output, usaremos también el de mistral.
        # Mantendremos la estructura para facilitar ampliaciones deseables a futuro.
        return respuesta_mistral(output_LLM)
        # return respuesta_ll2(output_LLM)

    elif config.modelo_activo == "cohere":
        return respuesta_cohere(output_LLM)

    else:
        return output_LLM


#  Limpieza previa de la respuesta del modelo independientemente de cual sea,
# para eliminar algunos de los "tics" habituales en outputs (salidas) de LLM.
def limpieza_generica_respuesta_modelo(output):
    print("\nLimpiando respuesta\n")
    # eliminamos la secuencia name_bot + ":" si está presente
    output = output.replace(config.name_bot + ":", "")

    # 1. Definimos un patrón de expresión regular para buscar la frase completa que contiene la palabra "Assistant"...
    pattern = re.compile(r'\bAssist\b.*?\.')
    # 2. ...y reemplazamos todas las coincidencias encontradas con una cadena vacía previo al nuevo output
    output = pattern.sub('', output)

    #  Se buscarán "<s>" y "</s>". Se eliminarán esas marcas sin afectar el resto del texto.
    # Los LLM a menudo introducen estas marcas, debido a parte del material original con el que han sido instruidos,
    # pero nosotros no queremos verlas en un chat.
    pattern = re.compile(r'</?s>')
    output = re.sub(pattern, '', output)

    #  Usamos una expresión regular para encontrar el primer signo de exclamación o interrogación.
    #  Esto es ideal para los modos asistente donde nos dan confirmación
    # a inicios de la respuesta, que en un chat automatizado es molesto de ver de manera repetida.
    confirmacion = re.search(r'[!?]', output)
    if confirmacion:
        # Si se encuentra un signo de exclamación o interrogación, verifica si hay un punto antes de este signo
        pre_text = output[:confirmacion.start()]
        if '.' not in pre_text:
            if len(pre_text) <= 15:
                output = output[confirmacion.start() + 1:]  # +1 para incluir el signo de puntuación en la eliminación

    return output


# Funcion tambien usable por llama 2 dada la similitud en su prompt y trato de respuesta.
def respuesta_mistral(output_LLM):  # metodo para limpiar las respuestas dadas por Mistral
    output_LLM = limpieza_generica_respuesta_modelo(output_LLM)
    #  se hace una primera limpieza del ouput en base al propio prompt del modelo,
    # que el LLM puede repetir en su respuesta
    index = output_LLM.find("[/INST]")
    if index != -1:  # se espera que mistral de la cadena de arriba en sus respuestas
        output_LLM = output_LLM[index + len("[/INST]"):]

    # limpiamos mas a fondo el output para quitarle strings innecesarios.
    output_LLM = limpieza_generica_respuesta_modelo(output_LLM)

    return output_LLM


#  Esta función podría eliminarse. Llama2 puede usar la función respuesta_mistral(output_LLM) sin problema.
# Por otra parte, la mantendremos como recordatorio si queremos algo más personalizado.
def respuesta_ll2(output_LLM):  # metodo para limpiar las respuestas dadas por Llama2
    output_LLM = limpieza_generica_respuesta_modelo(output_LLM)
    index = output_LLM.find("[/INST]")
    if index != -1:  # se espera que mistral de la cadena de arriba en sus respuestas
        return output_LLM[index + len("[/INST]"):]
    else:
        return output_LLM


# Modelo online: cohere.
def respuesta_cohere(output_LLM):  # metodo para limpiar las respuestas dadas por Cohere
    # las respuestas saldrán limpias, no se tratarán como los modelos offline de menor potencia.
    return output_LLM


def cargando_modelos():
    if config.modelo_activo == "mistral-7b":
        llm.myModel = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, device_map="auto",
                                                           trust_remote_code=False,
                                                           revision="main")
        llm.myTokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)

    # Se da el peculiar caso de que llama 2 y mistral comparten el mismo tipo de iniciado.
    # Se dejará aún así la parte de cada uno, a modo de referencia.
    elif config.modelo_activo == "llama2-7b":
        llm.myModel = AutoModelForCausalLM.from_pretrained(config.model_name_or_path,
                                                           device_map="auto",
                                                           trust_remote_code=False,
                                                           revision="main")

        llm.myTokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)

    elif config.modelo_activo == "cohere":
        llm.onlineModel = cohere.Client(cohere_key)


def mistral_7Bv2_LLM(prompt_template):
    # Generar respuesta del modelo.
    # Metodo 1 de inferencia. Es posible usar el que se ve con Llama 2 en este código, y viceversa,
    # siempre teniendo en cuenta la diferencia en parámetros para cada modelo.
    input_ids = llm.myTokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = llm.myModel.generate(
        inputs=input_ids,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        top_k=40,
        max_new_tokens=256,
        repetition_penalty=1.1  # Agrega este parámetro
    )

    generated_response = llm.myTokenizer.decode(output[0])
    print("salida Mistral 7b sin depurar:" + generated_response + "\n")
    return generated_response


def llama2_7B_chat_LLM(prompt_template):
    # Generar respuesta del modelo.
    # Metodo 2 de inferencia, usando pipeline. Es posible usar el que se ve con Mistral en este código, y viceversa,
    # siempre teniendo en cuenta la diferencia en parámetros para cada modelo.
    pipe = pipeline(
        "text-generation",
        model=llm.myModel,
        tokenizer=llm.myTokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.28  # llama 2 tiene ciertos problemas de repetitividad - parámetro experimental
    )
    generated_response = pipe(prompt_template)[0]['generated_text']
    print("salida llama 2 sin depurar:" + generated_response + "\n")
    return generated_response


def cohere_chat_LLM(prompt_template):
    # Enviar el prompt al modelo de Cohere y obtener la respuesta
    stream = llm.onlineModel.chat_stream(message=prompt_template,
                                         model='command',
                                         preamble="You are a friendly entity. Try to answer briefly with less than 50 "
                                                  + "words, and never exceed 100 words in your reply.",
                                         chat_history=llm.chat_log
                                         )
    chatbot_response = ""
    for event in stream:
        if event.event_type == "text-generation":
            print(event.text, end='')
            chatbot_response += event.text
    print("\n")
    return chatbot_response


def send_prompt(prompt):
    # Nuevo prompt
    formatted_prompt = build_prompt(prompt)
    print("Prompt con formato: " + formatted_prompt + "\n")

    # Entrada en el historial del prompt del usuario
    llm.chat_log.append({"role": config.name_user, "message": prompt})

    # Llamar a la función para obtener la respuesta del modelo
    config.output_LLM = output_by_model(formatted_prompt)
    print("\nSalida tal cual del output:" + config.output_LLM + "\n")

    config.output_LLM = respuesta_modelo(config.output_LLM)

    print("\nSalida sintetizada del modelo a raiz del output:" + config.output_LLM + "\n")

    # Entrada en el historial de la respuesta del LLM
    llm.chat_log.append({"role": config.name_bot, "message": config.output_LLM})

    # Se hace control de tokens para cuidar que no se pase de unos márgenes
    check_token_limit(llm.chat_log)

    config.prompt_en_proceso = False


def output_by_model(prompt):
    print("Output del modelo elegido:" + config.model_name_or_path + "\n")
    if config.modelo_activo == "mistral-7b":
        return mistral_7Bv2_LLM(prompt)

    elif config.modelo_activo == "llama2-7b":
        return llama2_7B_chat_LLM(prompt)

    elif config.modelo_activo == "cohere":
        return cohere_chat_LLM(prompt)

    else:
        return "1"
