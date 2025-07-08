import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import argparse

def visualizar_simulacao(csv_filename, speed=100):
    # Caminho relativo para o arquivo CSV
    csv_path = os.path.abspath(csv_filename)

    # Verificar se o arquivo existe antes de carregar
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")

    # Carregar os dados
    df = pd.read_csv(csv_path)

    # Criar coluna 'Channel' combinando R, G e B
    df['Channel'] = df.apply(lambda row: [row['R'] / 255, row['G'] / 255, row['B'] / 255], axis=1)

    # Gerar 'Frame' com base em 'Episode' e 'Step'
    df['Frame'] = df.groupby(['Episode', 'Step']).ngroup()

    # Determinar o tamanho do grid
    grid_width = df['Grid_Width'].iloc[0]
    grid_height = df['Grid_Height'].iloc[0]

    # Validar se todos os objetos estão dentro do grid
    if df['X'].max() > grid_width or df['Y'].max() > grid_height:
        raise ValueError("Alguns objetos estão fora dos limites do grid.")

    # Separar obstáculos (fixos) dos agentes (dinâmicos)
    obstacles = df[df['Type'] == 'Obstacle']
    agents = df[df['Type'] != 'Obstacle']

    # Configuração inicial da animação
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('dimgray')  # Fundo da figura
    ax.set_facecolor('black')  # Fundo do gráfico

    # Configurar os limites do gráfico com base no grid
    ax.set_xlim(-1, grid_width)
    ax.set_ylim(-1, grid_height)
    ax.margins(0)
    ax.axis('off')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)
    index_text_top = fig.text(0.5, 0.95, '', color='white', ha='center', va='top')

    # Scatter plot para obstáculos (fixos)
    scatter_obstacles = ax.scatter(obstacles['X'], obstacles['Y'], c=obstacles['Channel'].tolist(), s=30)

    # Inicializa scatter plots para agentes dinâmicos
    scatter_agents = {}
    for agent_type in agents['Type'].unique():
        scatter_agents[agent_type] = ax.scatter([], [], s=50)

    def init():
        # Inicializa os textos e os gráficos
        index_text_top.set_text('')
        for scatter in scatter_agents.values():
            scatter.set_offsets(np.empty((0, 2)))
        return [scatter_obstacles] + list(scatter_agents.values()) + [index_text_top]

    def update(frame):
        # Dados do frame atual
        current_df = df[df['Frame'] == frame]
        episode = current_df['Episode'].iloc[0]
        step = current_df['Step'].iloc[0]

        # Atualiza o texto do índice superior
        index_text_top.set_text(f'Episode: {episode}, Step: {step}')

        # Atualiza agentes dinâmicos
        for agent_type, scatter in scatter_agents.items():
            # Apenas agentes vivos (Is_Alive=True) são exibidos
            group = current_df[(current_df['Type'] == agent_type) & (current_df['Is_Alive'] == True)]
            scatter_coords = group[['X', 'Y']].values if not group.empty else np.empty((0, 2))
            scatter_colors = group['Channel'].tolist() if not group.empty else []
            scatter.set_offsets(scatter_coords)
            scatter.set_color(scatter_colors)
            scatter.set_visible(True)

        return [scatter_obstacles] + list(scatter_agents.values()) + [index_text_top]

    # Criar animação
    ani = FuncAnimation(fig, update, frames=df['Frame'].unique(), init_func=init, interval=speed)

    plt.show()



def preparar_e_chamar_visualizacao(nome_do_arquivo_csv, speed=100):
    if not nome_do_arquivo_csv:
        print("Erro: Nenhum nome de arquivo CSV fornecido.")
        return

    # Chama a função de visualização diretamente
    visualizar_simulacao(nome_do_arquivo_csv, speed=speed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizar a simulação de agentes a partir de um arquivo CSV.')
    parser.add_argument('--speed', type=int, default=100, help='Intervalo entre frames em milissegundos (maior = mais lento)')
    parser.add_argument('csv_filename', type=str, help='Caminho para o arquivo CSV a ser usado na simulação.')
    args = parser.parse_args()

    preparar_e_chamar_visualizacao(args.csv_filename, speed=args.speed)
