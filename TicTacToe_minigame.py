import sys
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QTextBrowser, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt   # Gestion textos en espacios

class TicTacToe(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tic Tac Toe")
        self.setGeometry(100, 100, 800, 350)  # Ajusta el tamaño de la ventana para dar espacio al conjunto

        self.central_widget = QFrame()
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Lado izquierdo
        self.left_frame = QFrame()
        self.left_frame.setFixedWidth(100)
        self.layout.addWidget(self.left_frame)

        self.left_layout = QVBoxLayout()
        self.left_frame.setLayout(self.left_layout)

        self.grid_size_combo_box = QComboBox()
        self.grid_size_combo_box.addItem("3x3")
        self.grid_size_combo_box.addItem("4x4")
        self.grid_size_combo_box.currentIndexChanged.connect(self.change_grid_size)
        self.left_layout.addWidget(self.grid_size_combo_box)

        self.change_difficulty = QComboBox()
        self.change_difficulty.addItem("Normal")
        self.change_difficulty.addItem("Retante")
        self.change_difficulty.currentIndexChanged.connect(self.change_difficulty_level)
        self.left_layout.addWidget(self.change_difficulty)

        self.new_game_button = QPushButton("Inicio Juego")
        self.new_game_button.clicked.connect(self.init_game)
        self.left_layout.addWidget(self.new_game_button)

        # Centro - Área del juego
        self.center_frame = QFrame()
        self.center_frame.setStyleSheet("background-color: white;")
        self.layout.addWidget(self.center_frame)

        self.center_layout = QVBoxLayout()
        self.center_frame.setLayout(self.center_layout)

        self.frame_area_game = QFrame()
        self.center_layout.addWidget(self.frame_area_game)

        # Lado derecho - QTextBrowser
        self.right_frame = QFrame()
        self.right_frame.setFixedWidth(200)
        self.layout.addWidget(self.right_frame)

        self.right_layout = QVBoxLayout()
        self.right_frame.setLayout(self.right_layout)

        self.info_browser = QTextBrowser()
        self.info_browser.setFont(QFont("Input Sans", 14))
        self.info_browser.setStyleSheet(
            "QTextBrowser { background-color: white; color: black; }")
        self.info_browser.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(self.info_browser)

        self.player_wins = 0
        self.ai_wins = 0
        self.turn_count = 0  # Contador de turnos

        self.init_game()

    def init_game(self):
        self.turno = 1  # Player 1 (jugador) inicio
        self.board_size = int(self.grid_size_combo_box.currentText()[0])
        self.board = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.turn_count = 0  # Reiniciar el contador de turnos

        self.info_browser.setText(f"El jugador es (X) \nJugador: {self.player_wins}, Máquina: {self.ai_wins}")

        self.frame_area_game.deleteLater()
        self.frame_area_game = QFrame()
        self.frame_area_game.setStyleSheet("background-color: white;")
        self.center_layout.addWidget(self.frame_area_game)

        self.buttons = []
        grid_layout = QVBoxLayout()
        for i in range(self.board_size):
            row_layout = QHBoxLayout()
            for j in range(self.board_size):
                button = QPushButton()
                button.setFixedSize(60, 60)
                button.setStyleSheet("background-color: lightblue;")
                # Capturamos coordenadas con lambda al clickar
                button.clicked.connect(lambda _, x=i, y=j: self.button_clicked(x, y))
                row_layout.addWidget(button)
                self.buttons.append(button)
            grid_layout.addLayout(row_layout)
        self.frame_area_game.setLayout(grid_layout)

        # el jugador 2 / turno par, será la IA
        if self.turno == 2:
            self.ai_move()

    def button_clicked(self, x, y):
        if self.board[x][y] is None and self.turno == 1:
            self.buttons[x * self.board_size + y].setStyleSheet("background-color: darkblue;")
            self.buttons[x * self.board_size + y].setFont(QFont('Arial Black', 24))
            self.buttons[x * self.board_size + y].setText("X")
            self.buttons[x * self.board_size + y].setStyleSheet("color: lightblue;")
            self.board[x][y] = "X"
            if self.check_winner("X"):
                self.player_wins += 1
                self.info_browser.setText(f"¡Has ganado! \nJugador: {self.player_wins}, Máquina: {self.ai_wins}")
                self.disable_buttons()
            elif self.check_draw():
                self.info_browser.setText("¡Empate!")
                self.disable_buttons()
            else:
                self.turno = 2
                self.turn_count += 1  # Incrementar el contador de turnos
                self.info_browser.setText(f"Turno de la máquina (O) \nJugador: {self.player_wins}, Máquina: {self.ai_wins}")
                self.ai_move()

    def ai_move(self):  # Elegirá entre dos dificultades, según lo elegido en el combobox.
        if self.turno == 2:
            if self.change_difficulty.currentText() == "Retante":  # La poda será menos intensa en tableros grandes
                # Incrementar la profundidad dependiendo del número de turnos
                self.poda_profundidad = min(2 + self.turn_count // 2, 9) if self.board_size == 3 else min(2 + self.turn_count // 2, 6)
                x, y = self.minimax(self.board, self.poda_profundidad, -float('inf'), float('inf'), True)[1]
            else:  # seleccionará meramente una casilla disponible al azar
                movimientos_posibles = [(i, j) for i in range(self.board_size) for j in range(self.board_size) if self.board[i][j] is None]
                if movimientos_posibles:
                    x, y = random.choice(movimientos_posibles)

            self.buttons[x * self.board_size + y].setStyleSheet("background-color: darkred;")
            self.buttons[x * self.board_size + y].setFont(QFont('Arial Black', 24))
            self.buttons[x * self.board_size + y].setText("O")
            self.buttons[x * self.board_size + y].setStyleSheet("color: lightcoral;")
            self.board[x][y] = "O"
            if self.check_winner("O"):
                self.ai_wins += 1
                self.info_browser.setText(f"¡La máquina ha ganado! \nJugador: {self.player_wins}, Máquina: {self.ai_wins}")
                self.disable_buttons()
            elif self.check_draw():
                self.info_browser.setText("¡Empate!")
                self.disable_buttons()
            else:
                self.turno = 1
                self.turn_count += 1  # Incrementar el contador de turnos
                self.info_browser.setText(f"El jugador es (X) \nJugador: {self.player_wins}, Máquina: {self.ai_wins}")

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        if self.check_winner("O"):
            return 1, None
        elif self.check_winner("X"):
            return -1, None
        elif self.check_draw():
            return 0, None

        if depth == 0:
            return 0, None

        if is_maximizing:
            # Inicialización para prepararlo para el Minimax que usamos: mayor a negativo infinito.
            # Inicia la evaluación máxima.
            max_eval = -float('inf')

            mejor_movimiento = None
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board[i][j] is None:
                        board[i][j] = "O"
                        eval = self.minimax(board, depth - 1, alpha, beta, False)[0]
                        board[i][j] = None
                        if eval > max_eval:
                            max_eval = eval
                            mejor_movimiento = (i, j)
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
            return max_eval, mejor_movimiento
        else:
            # Inicialización para prepararlo para el Minimax que usamos: menor a positivo infinito.
            # Inicia la evaluación mínima.
            min_eval = float('inf')
            mejor_movimiento = None

            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board[i][j] is None:
                        board[i][j] = "X"
                        eval = self.minimax(board, depth - 1, alpha, beta, True)[0]
                        board[i][j] = None
                        if eval < min_eval:
                            min_eval = eval
                            mejor_movimiento = (i, j)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
            return min_eval, mejor_movimiento

    def check_winner(self, player):  # Iteración para ver si hay ganador. Mirará todas las formas de ganar.
        for i in range(self.board_size):
            if all(self.board[i][j] == player for j in range(self.board_size)):
                return True
            if all(self.board[j][i] == player for j in range(self.board_size)):
                return True
        if all(self.board[i][i] == player for i in range(self.board_size)):
            return True
        if all(self.board[i][self.board_size - i - 1] == player for i in range(self.board_size)):
            return True
        return False

    def check_draw(self):
        return all(self.board[i][j] is not None for i in range(self.board_size) for j in range(self.board_size))

    def disable_buttons(self):
        for button in self.buttons:
            button.setEnabled(False)

    def change_grid_size(self):
        self.init_game()

    def change_difficulty_level(self):
        self.turn_count = 0  # Reiniciar el contador de turnos al cambiar la dificultad
        if self.change_difficulty.currentText() == "Retante":
            if self.grid_size_combo_box.currentText() == "3x3":
                self.poda_profundidad = 2  # Profundidad inicial baja
            elif self.grid_size_combo_box.currentText() == "4x4":
                self.poda_profundidad = 2  # Profundidad inicial baja
        else:
            self.poda_profundidad = 9


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TicTacToe()
    window.show()
    sys.exit(app.exec_())
