import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt

import LLM_manager
from main_interface import Ui_MainWindow
from utils import send_chat_to_gui
from chat_config import ChatConfig


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.config = ChatConfig()  # Usar chat_config.variable para acceso

        # Recoger datos modelo por defecto y sacarlos por textBrowserStatus
        LLM_manager.my_model()
        self.ui.update_pantalla_status()

        self.ui.pushButtonSend.clicked.connect(self.send_input_manual)
        # Enter en lineEdit1 equivale a darle al botón Enviar
        self.ui.lineEditInput.returnPressed.connect(self.send_input_manual)

        # Se necesita una referencia al prompt para almacenarlo
        self.prompt_user = ""

    def show_options_tab(self):
        self.ui.tabWidget.setCurrentWidget(self.ui.tabOptions)

    def show_main_tab(self):
        self.ui.tabWidget.setCurrentWidget(self.ui.tabMain)

    def send_input_manual(self):
        input_value = self.ui.lineEditInput.text()
        # Guardar el prompt ingresado por el usuario
        config.prompt_user = input_value
        LLM_manager.send_prompt(input_value)

        # se limpia linea de input
        self.ui.lineEditInput.clear()

        # Después de enviar el prompt, llama al método para imprimir el valor de output_LLM
        send_chat_to_gui("manual", config.output_LLM, config.prompt_user, self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    LLM_manager.my_model()
    LLM_manager.cargando_modelos()
    window = MainWindow()
    window.show()
    config = ChatConfig()

    # estas 2 lineas gestionarán un inicio en el chat
    LLM_manager.send_prompt(config.prompt_begin)
    send_chat_to_gui("intro", config.output_LLM, config.prompt_begin, window)

    # inicia el chat automatizado del modelo actual
    LLM_manager.start_timer(window)  # Inicia temporizador y pasa instancia ventana principal
    sys.exit(app.exec_())
