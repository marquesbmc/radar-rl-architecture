import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ast import literal_eval
import os
from PyQt5.QtGui import QIcon
import threading

# Função de visualização principal
def visualizar_simulacao(csv_filename='sim_orcasafricans2.csv'):
    # Caminho relativo para o arquivo CSV na pasta /save/sim
    csv_path = os.path.join(os.path.dirname(__file__), 'save', 'sim', csv_filename)


    # Verificar se o arquivo existe antes de carregar
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")

    # Carregar os dados
    df = pd.read_csv(csv_path)

    # Carregar os dados
    df = pd.read_csv(csv_path)

    # Preparar os dados
    df['Channel'] = df['Channel'].apply(literal_eval)
    df[['X', 'Y']] = df['Position'].str.extract(r'\((\d+), (\d+)\)').astype(int)

    # Configuração inicial da animação
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('dimgray')
    ax.set_facecolor('black')

    ax.set_xlim(df['X'].min() - 1, df['X'].max() + 1)
    ax.set_ylim(df['Y'].min() - 1, df['Y'].max() + 1)
    ax.margins(0)
    ax.axis('off')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)
    index_text_top = fig.text(0.5, 0.95, '', color='white', ha='center', va='top')
    scatters = {}

    processed_filename = csv_filename.removeprefix("sim_").removesuffix(".csv")


    def init():
        for name in df['Name'].unique():
            scatters[name] = ax.scatter([], [], label=name, color='white')
        return list(scatters.values()) + [index_text_top]

    def update(frame):
        current_df = df[df['Index'] == frame]
        episode = current_df['Episode'].iloc[0]
        step = current_df['Step'].iloc[0]
        index_text_top.set_text(f'Model: {processed_filename} - Episode: {episode}, Step: {step}')

        for name, group in current_df.groupby('Name'):
            is_alive = group['Is Alive'].iloc[-1]
            rgba_color = np.array(group['Channel'].iloc[-1])

            if not is_alive:
                scatters[name].set_visible(False)
            else:
                scatters[name].set_offsets([group[['X', 'Y']].values[-1]])
                scatters[name].set_color(rgba_color)
                scatters[name].set_visible(True)

        return list(scatters.values()) + [index_text_top]

    # Definir o ícone da janela
    icon_path = os.path.join(os.path.dirname(__file__), 'images', 'icon.png')
    try:
        if os.path.exists(icon_path):
            plt.get_current_fig_manager().window.setWindowIcon(QIcon(icon_path))
        else:
            print(f"Icon not found in path: {icon_path}")
    except AttributeError:
        print("Changing the icon may not be supported by the current backend.")

    # Definir o título da janela
    try:
        plt.get_current_fig_manager().set_window_title('Simulation Monitor')
    except AttributeError:
        print("Changing the window title may not be supported by the current backend.")

    # Criar animação
    ani = FuncAnimation(fig, update, frames=df['Index'].unique(), init_func=init, interval=1)

    plt.show()

# Função que executa a preparação dos dados em uma thread separada
def preparar_e_chamar_visualizacao(nome_do_arquivo_csv):
    if not nome_do_arquivo_csv:
        print("Erro: Nenhum nome de arquivo CSV fornecido.")
        return
    
    # Criar uma thread para preparar os dados e sinalizar a execução da visualização
    threading.Thread(target=lambda: visualizar_simulacao(nome_do_arquivo_csv)).start()
