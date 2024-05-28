# chat_config.py
# import prototipo_Main
from transformers import AutoModelForCausalLM, AutoTokenizer

# from LLM_manager import config


class ChatConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.name_user = 'Usuario'
            cls._instance.name_bot = 'UOCBot'
            cls._instance.prompt_user = "Tell me something."
            cls._instance.output_LLM = ""
            cls._instance.prompt_begin = "Tell me some random fact about anything."
            cls._instance.prompt_auto = "Tell me more about that."
            # cls._instance.current_speak_thread = None
            cls._instance.timer_index_gui = 2  # [0] = 15 segundos para testeos. Recom.minimo [2] = 1 minuto.
            cls._instance.timer_auto_llm = 30  # temporizador modo automatico - cambiable desde GUI vinculada a index
            cls._instance.llm_index_gui = 0  # [0] LOCAL Mistral 7B Instruct 0.2 [1] LOCAL LLAMA 2 7B Chat [2] ONLINE Cohere
            cls._instance.context_index = 2  # Tokens contexto: [0] 1024 [1] 2048 [2] 4096
            cls._instance.context_size = 4096  # Tokens contexto cambiables desde GUI vinculada a index
            # cls._instance.chat_history = []  # Lista para almacenar los mensajes del chat
            cls._instance.text2speech_index = 0  # [0] Desactivado [1] Local: Inglés [2] Online: Español (google T2S)
            cls._instance.model_name_or_path = ""
            cls._instance.modelo_activo = "llama2-7b"  # Ejemplo de uso.
            cls._instance.modelo_licencia = ""
            cls._instance.make_auto_llm_restart = False  # para reinicio reloj modo automatizado llm
            cls._instance.stop_llm = False
            cls._instance.prompt_en_proceso = False
        return cls._instance




