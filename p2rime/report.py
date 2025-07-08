import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile

from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from tkinter import messagebox
from PIL import Image
from datetime import datetime



def load_json_data(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        messagebox.showerror("Error", f"Failed to load JSON file: {e}")
        return None

def draw_subtitle_with_tooltip(canvas_obj, x, y, text, tooltip_text):
    canvas_obj.setFont("Helvetica-Bold", 14)
    canvas_obj.setFillColor(colors.black)
    canvas_obj.drawString(x, y, text)
    canvas_obj.setStrokeColor(colors.lightgrey)
    canvas_obj.setLineWidth(1.2)
    canvas_obj.line(x, y - 5, x + 500, y - 5)

def read_simulation_data_report(data):
    if 'quantitative_data' not in data:
        print("No quantitative data found in the provided JSON.")
        return pd.DataFrame(), pd.DataFrame()

    predator_df = pd.DataFrame(data.get('quantitative_data', {}).get('predator', []), columns=['Reward', 'Done', 'Step'])
    prey_df = pd.DataFrame(data.get('quantitative_data', {}).get('prey', []), columns=['Reward', 'Done', 'Step'])

    return predator_df, prey_df

def calculate_statistics_report(data):
    if len(data) == 0:
        return {
            'Mean': '',
            'Median': '',
            'SD': '',
            'Max': '',
            'Min': '',
            'Variance': '',
            'Range': '',
            'IRQ': ''
        }

    return {
        'Mean': np.mean(data),
        'Median': np.median(data),
        'SD': np.std(data),
        'Max': np.max(data),
        'Min': np.min(data),
        'Variance': np.var(data),
        'Range': np.max(data) - np.min(data),
        'IRQ': np.percentile(data, 75) - np.percentile(data, 25),
    }

def create_quantitative_report(data):
    predator_df, prey_df = read_simulation_data_report(data)

    predator_stats = {
        'Reward': calculate_statistics_report(predator_df['Reward']),
        'Done': calculate_statistics_report(predator_df['Done']),
        'Step': calculate_statistics_report(predator_df['Step']),
    }

    prey_stats = {
        'Reward': calculate_statistics_report(prey_df['Reward']),
        'Done': calculate_statistics_report(prey_df['Done']),
        'Step': calculate_statistics_report(prey_df['Step']),
    }

    return predator_stats, prey_stats

def add_title_section(c, data, width, height):
    # Adiciona a imagem do logo antes de qualquer outra coisa, reduzindo o tamanho em 20%
    logo_path = "images/logo.png"  # Certifique-se de que esse caminho está correto e que a imagem está no local certo

    # Obtém as dimensões originais da imagem
    with Image.open(logo_path) as img:
        original_width, original_height = img.size

    # Calcula o novo tamanho, reduzindo em 20%
    new_width = original_width * 0.8
    new_height = original_height * 0.8

    # Adiciona a imagem redimensionada ao PDF
    c.drawImage(logo_path, 40, height - 80, width=new_width, height=new_height, mask='auto')

    # Aproxima ainda mais o título do relatório da figura, colocando-o imediatamente abaixo
    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, height - 55 - new_height, "Model Simulation Report")

    # Adiciona a data atual bem próxima do título
    current_date = datetime.now().strftime("%Y-%m-%d")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 105 - new_height + 5, "Date: {}".format(current_date))

    # Adiciona o nome do modelo logo abaixo da data
    model_name = data.get('model', 'Unknown Model')
    c.drawString(40, height - 85 - new_height + 5, "Model Name: {}".format(model_name))

def add_footer(c, model_name, creation_date, page_number, total_pages):
    """Adiciona um rodapé na página do PDF com o nome do modelo, data de criação, e número da página."""
    footer_text = f"Model: {model_name} | Date: {creation_date} | Page: {page_number}/{total_pages}"
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.grey)
    c.drawCentredString(A4[0] / 2, 20, footer_text)

def generate_pdf_report(data, output_pdf_path):
    # Configuração do Canvas
    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    width, height = A4

    # Definições de espaçamento e tamanho de fonte
    line_spacing = 15  # Espaçamento entre linhas de texto
    section_spacing = 30  # Espaçamento entre seções
    title_to_content_spacing = 30  # Espaçamento entre o título da seção e o conteúdo abaixo
    subtitle_font_size = 12  # Tamanho dos subtítulos das seções
    text_font_size = 10  # Tamanho do texto do conteúdo

    # Pegando o nome do modelo e data de criação para o rodapé
    model_name = data.get('model', 'Unknown Model')
    creation_date = datetime.now().strftime("%Y-%m-%d")
    total_pages = 3

    # Primeira página - Título, Logo e algumas seções
    add_title_section(c, data, width, height)
    y = height - 145

    y -= section_spacing
    y = add_environment_section(c, data, y, line_spacing, title_to_content_spacing, subtitle_font_size, text_font_size)

    y -= section_spacing
    y = add_population_section(c, data, y, line_spacing, title_to_content_spacing, subtitle_font_size, text_font_size)

    y -= section_spacing
    y = add_neural_network_section(c, data, y, line_spacing, title_to_content_spacing, subtitle_font_size, text_font_size)

    # Adiciona o rodapé e avança para a próxima página
    add_footer(c, model_name, creation_date, page_number=1, total_pages=total_pages)
    c.showPage()  # Finaliza a primeira página

    # Segunda página - Quantitative Data
    predator_stats, prey_stats = create_quantitative_report(data)
    y = height - 100 + 40 + 20
    y = add_quantitative_data_section(c, predator_stats, prey_stats, width, y, line_spacing, title_to_content_spacing, subtitle_font_size, text_font_size)

    y -= section_spacing
    y = add_behavior_data_section(c, data, y, line_spacing, title_to_content_spacing, subtitle_font_size, text_font_size)

    # Adiciona o rodapé e avança para a próxima página
    add_footer(c, model_name, creation_date, page_number=2, total_pages=total_pages)
    c.showPage()  # Finaliza a segunda página

    # Terceira página - Gráficos da Simulação
    y = height - 100 + 40 + 20
    y = add_simulation_data_chart_section(c, data, y, line_spacing, title_to_content_spacing, subtitle_font_size, text_font_size)

    # Adiciona o rodapé na última página
    add_footer(c, model_name, creation_date, page_number=3, total_pages=total_pages)

    # Salva o PDF
    c.save()
    messagebox.showinfo("PDF Generated", f"The PDF has been saved as '{output_pdf_path}'.")


def add_environment_section(c, data, y, line_spacing, title_to_content_spacing, subtitle_font_size, text_font_size):
    # Configura o título da seção
    c.setFont("Helvetica-Bold", subtitle_font_size)
    draw_subtitle_with_tooltip(c, 40, y, "Environment", "Defines the simulation environment details such as grid size and episode settings.")
    
    # Espaçamento entre o título da seção e o conteúdo
    y -= title_to_content_spacing

    # Configura o texto do conteúdo
    c.setFont("Helvetica", text_font_size)

    # Tamanho do Grid
    c.drawString(60, y, "Grid Side: Defines the size of one side of the grid, making it a square.")
    y -= line_spacing  # Ajusta a posição vertical para a próxima linha
    c.drawString(80, y, f"{data['environment']['grid_size']}")

    # Episódios
    y -= line_spacing
    c.drawString(60, y, "Episodes: Total number of episodes to be executed in the simulation.")
    y -= line_spacing  # Ajusta a posição vertical para a próxima linha
    c.drawString(80, y, f"{data['environment']['episodes']}")

    # Passos por Episódio
    y -= line_spacing
    c.drawString(60, y, "Steps per Episode: Number of steps allowed in each simulation episode.")
    y -= line_spacing  # Ajusta a posição vertical para a próxima linha
    c.drawString(80, y, f"{data['environment']['steps']}")

    return y - line_spacing  # Retorna a posição vertical final para continuar o fluxo do relatório

def add_population_section(c, data, y, line_spacing, title_to_content_spacing, subtitle_font_size, text_font_size):
    # Configura o título da seção
    c.setFont("Helvetica-Bold", subtitle_font_size)
    draw_subtitle_with_tooltip(c, 40, y, "Population", "Population details including initial counts and spawn rates.")
    
    # Espaçamento entre o título da seção e o conteúdo
    y -= title_to_content_spacing  

    # Configura o texto do conteúdo
    c.setFont("Helvetica", text_font_size)

    # Contagem Inicial
    c.drawString(60, y, "Initial Count: Specifies the initial number of agents at the start of the simulation.")
    y -= line_spacing  # Ajusta a posição vertical para a próxima linha
    c.drawString(80, y, f"Predator: {data['population']['predator_initial_count']}")
    y -= line_spacing  # Ajusta a posição vertical para a próxima linha
    c.drawString(80, y, f"Prey: {data['population']['prey_initial_count']}")

    # Contagem Máxima
    y -= line_spacing
    c.drawString(60, y, "Max Count: Defines the maximum number of agents allowed in the simulation.")
    y -= line_spacing  # Ajusta a posição vertical para a próxima linha
    c.drawString(80, y, f"Predator: {data['population']['predator_max_count']}")
    y -= line_spacing  # Ajusta a posição vertical para a próxima linha
    c.drawString(80, y, f"Prey: {data['population']['prey_max_count']}")

    # Taxa de Reprodução
    y -= line_spacing
    c.drawString(60, y, "Spawn Rate: Percentage chance of agent reproduction.")
    y -= line_spacing  # Ajusta a posição vertical para a próxima linha
    c.drawString(80, y, f"Predator: {data['population']['predator_spawn_rate']}%")
    y -= line_spacing  # Ajusta a posição vertical para a próxima linha
    c.drawString(80, y, f"Prey: {data['population']['prey_spawn_rate']}%")

    # Decaimento por Etapa
    y -= line_spacing
    c.drawString(60, y, "Step Decay: Amount of life points lost by agents for each step taken.")
    y -= line_spacing  # Ajusta a posição vertical para a próxima linha
    c.drawString(80, y, f"Predator: {data['population']['predator_step_decay']}%")
    y -= line_spacing  # Ajusta a posição vertical para a próxima linha
    c.drawString(80, y, f"Prey: {data['population']['prey_step_decay']}%")

    return y - line_spacing  # Retorna a posição vertical final para continuar o fluxo do relatório

def add_neural_network_section(c, data, y, line_spacing, title_to_content_spacing, subtitle_font_size, text_font_size):
    # Configura o título da seção
    c.setFont("Helvetica-Bold", subtitle_font_size)
    draw_subtitle_with_tooltip(c, 40, y, "Neural Network", "Neural network configurations for prey and predator agents.")
    
    # Espaçamento entre o título da seção e o conteúdo
    y -= title_to_content_spacing  

    # Configura o texto do conteúdo
    c.setFont("Helvetica", text_font_size)

    # Modelo de Aprendizado
    c.drawString(60, y, "Learning Model: Specifies the type of learning model used by predator and prey agents in the simulation")  
    y -= line_spacing  # Ajusta a posição vertical para a próxima linha
    c.drawString(80, y, f"Predator: {data['neural_network']['predator_learning_model']}")
    y -= line_spacing  # Ajusta a posição vertical para a próxima linha
    c.drawString(80, y, f"Prey: {data['neural_network']['prey_learning_model']}")

    # Camada Avançada
    y -= line_spacing
    c.drawString(60, y, "Advanced Layer: Specifies any additional layers or network modifications used to enhance agent learning.")
    y -= line_spacing  # Ajusta a posição vertical para a próxima linha
    c.drawString(80, y, f"Predator: {data['neural_network']['predator_design_variant']}")
    y -= line_spacing  # Ajusta a posição vertical para a próxima linha
    c.drawString(80, y, f"Prey: {data['neural_network']['prey_design_variant']}")


    return y - line_spacing  # Retorna a posição vertical final para continuar o fluxo do relatório

def add_quantitative_data_section(c, predator_stats, prey_stats, width, y, line_spacing, title_to_content_spacing, subtitle_font_size, text_font_size):
    # Título da Seção com Subtítulo e Tooltip
    c.setFont("Helvetica-Bold", subtitle_font_size)
    draw_subtitle_with_tooltip(c, 40, y, "Quantitative Population Data", "Summary statistics of agent populations.")
    
    # Espaçamento entre o título da seção e o conteúdo
    y -= title_to_content_spacing  

    # Configura o texto do conteúdo
    c.setFont("Helvetica-Bold", text_font_size)

    # Configura a largura das colunas para acomodar três métricas (Reward, Done, Step)
    column_width = (width - 120) / 4  # Deixa uma margem de 60 em cada lado




    # Criando a tabela para os dados de Predador
    c.drawString(2 * column_width, y, "Predator Stats")

    # Configura o texto do conteúdo
    c.setFont("Helvetica", text_font_size)
    
    # Títulos das colunas para Predator Stats
    y -= line_spacing  # Ajusta a posição para os títulos das colunas
    c.drawString(60, y, "")  # Espaço para a coluna de métricas
    c.drawString(60 + column_width, y, "Reward")
    c.drawString(60 + column_width * 2, y, "Done")
    c.drawString(60 + column_width * 3, y, "Step")
    
    # Métricas para Predator
    metrics = ['Mean', 'Median', 'SD', 'Max', 'Min', 'Variance', 'Range', 'IRQ']  # Métricas estatísticas
    for i, metric in enumerate(metrics):
        y -= line_spacing
        c.drawString(60, y, metric)  # Nome da métrica
        c.drawString(60 + column_width, y, str(predator_stats['Reward'][metric]))  # Valor para 'Reward'
        c.drawString(60 + column_width * 2, y, str(predator_stats['Done'][metric]))  # Valor para 'Done'
        c.drawString(60 + column_width * 3, y, str(predator_stats['Step'][metric]))  # Valor para 'Step'

    # Espaço antes da tabela de Prey Stats
    y -= line_spacing * 2

    # Configura o texto do conteúdo
    c.setFont("Helvetica-Bold", text_font_size)

    # Criando a tabela para os dados de Prey
    c.drawString(2 * column_width, y, "Prey Stats")

    # Configura o texto do conteúdo
    c.setFont("Helvetica", text_font_size)
    
    # Títulos das colunas para Prey Stats
    y -= line_spacing
    c.drawString(60, y, "")  # Espaço para a coluna de métricas
    c.drawString(60 + column_width, y, "Reward")
    c.drawString(60 + column_width * 2, y, "Done")
    c.drawString(60 + column_width * 3, y, "Step")

    # Métricas para Prey
    for i, metric in enumerate(metrics):
        y -= line_spacing
        c.drawString(60, y, metric)  # Nome da métrica
        c.drawString(60 + column_width, y, str(prey_stats['Reward'][metric]))  # Valor para 'Reward'
        c.drawString(60 + column_width * 2, y, str(prey_stats['Done'][metric]))  # Valor para 'Done'
        c.drawString(60 + column_width * 3, y, str(prey_stats['Step'][metric]))  # Valor para 'Step'
    
    return y - line_spacing  # Retorna a posição final para a próxima seção

def add_behavior_data_section(c, data, y, line_spacing, title_to_content_spacing, subtitle_font_size, text_font_size):
    # Configura o título da seção
    c.setFont("Helvetica-Bold", subtitle_font_size)
    draw_subtitle_with_tooltip(c, 40, y, "Behavior Data", "Behavioral statistics of predator and prey agents during simulation.")
    
    # Espaçamento entre o título da seção e o conteúdo
    y -= title_to_content_spacing  

    # Configura o texto do conteúdo
    c.setFont("Helvetica-Bold", text_font_size)

    # Define a largura das colunas e desenha os títulos das colunas
    left_column_x = 60
    right_column_x = 300
    row_height = y

    # Títulos das colunas
    c.drawString(left_column_x, row_height, "Predator Behavior Stats")
    c.drawString(right_column_x, row_height, "Prey Behavior Stats")
    row_height -= line_spacing

    c.setFont("Helvetica", text_font_size)

    # Dados de Predator e Prey lado a lado
    c.drawString(left_column_x, row_height, "Prey Captured:")
    c.drawString(left_column_x + 120, row_height, str(data['behavior_data']['predator']['Prey captured']))
    c.drawString(right_column_x, row_height, "Predator Escape:")
    c.drawString(right_column_x + 120, row_height, str(data['behavior_data']['prey']['Predator escape']))
    row_height -= line_spacing

    c.drawString(left_column_x, row_height, "Nearby Prey:")
    c.drawString(left_column_x + 120, row_height, str(data['behavior_data']['predator']['Nearby prey']))
    c.drawString(right_column_x, row_height, "Nearby Predator:")
    c.drawString(right_column_x + 120, row_height, str(data['behavior_data']['prey']['Nearby predator']))
    row_height -= line_spacing

    c.drawString(left_column_x, row_height, "Exploring Map:")
    c.drawString(left_column_x + 120, row_height, str(data['behavior_data']['predator']['Exploring map']))
    c.drawString(right_column_x, row_height, "Exploring Map:")
    c.drawString(right_column_x + 120, row_height, str(data['behavior_data']['prey']['Exploring map']))

    # Retorna a posição final após a tabela
    return row_height - line_spacing * 2

def create_simulation_data_dataframe(data):
    # Extrai a lista de dados da chave `simulation_data`
    simulation_data = data.get('simulation_data', [])
    
    # Cria o DataFrame com colunas específicas para Episode, Prey e Predator
    df = pd.DataFrame(simulation_data, columns=['Episode', 'Prey', 'Predator'])
    
    return df

def add_simulation_data_chart_section(c, data, y, line_spacing, title_to_content_spacing, subtitle_font_size, text_font_size):
    # Cria o DataFrame a partir dos dados de `simulation_data`
    simulation_df = create_simulation_data_dataframe(data)

    # Define a posição inicial no topo da página
    y = c._pagesize[1] - 40

    # Configura o título da seção
    c.setFont("Helvetica-Bold", subtitle_font_size)
    draw_subtitle_with_tooltip(c, 40, y, "Population Charts", "Various visualizations of prey and predator population dynamics.")
    
    # Adiciona um espaçamento mais amplo após o título para garantir que os gráficos fiquem abaixo
    y -= (title_to_content_spacing + line_spacing * 8)  # Espaçamento extra aumentado

    # Define o tamanho dos gráficos e posições padronizadas
    graph_width, graph_height = 250, 125
    x_positions = [40, 300]  # Posições x para alinhar dois gráficos lado a lado
    vertical_gap = graph_height + line_spacing * 1 # Aumenta o espaçamento fixo entre linhas de gráficos

    # Lista de informações de gráficos para Prey e Predator
    chart_info = [
        ('Prey', 'b', "Prey Count per Episode"),
        ('Predator', 'r', "Predator Count per Episode")
    ]

    # Ajuste a posição y para os gráficos de episódio vs. população
    for i, (y_column, color, title) in enumerate(chart_info):
        # Gráfico original com título no próprio gráfico
        chart_path = generate_population_episode_chart(simulation_df, y_column, color, title, show_moving_average=False)
        c.drawImage(chart_path, x_positions[0], y - (i * vertical_gap), width=graph_width, height=graph_height)
        os.remove(chart_path)

        # Gráfico com média móvel com título no próprio gráfico
        chart_path_moving_avg = generate_population_episode_chart(simulation_df, y_column, color, title + " (Moving Average 25%)", show_moving_average=True)
        c.drawImage(chart_path_moving_avg, x_positions[1], y - (i * vertical_gap), width=graph_width, height=graph_height)
        os.remove(chart_path_moving_avg)

    # Gráfico combinado de Prey e Predator na terceira linha
    y -= vertical_gap * 2  # Ajusta y para uma nova linha de gráficos
    combined_title = "Prey and Predator Count per Episode"
    chart_path_combined = generate_comparison_chart(simulation_df, combined_title, show_moving_average=False)
    chart_path_combined_moving_avg = generate_comparison_chart(simulation_df, combined_title + " (Moving Average 25%)", show_moving_average=True)

    c.drawImage(chart_path_combined, x_positions[0], y, width=graph_width, height=graph_height)
    os.remove(chart_path_combined)

    c.drawImage(chart_path_combined_moving_avg, x_positions[1], y, width=graph_width, height=graph_height)
    os.remove(chart_path_combined_moving_avg)

    # Gráfico de Prey vs Predator (scatter com regressão linear) na próxima linha
    y -= vertical_gap
    scatter_title = "Prey vs Predator Population"
    chart_path_prey_vs_predator = generate_prey_vs_predator_chart(simulation_df, scatter_title, show_moving_average=False)
    chart_path_prey_vs_predator_moving_avg = generate_prey_vs_predator_chart(simulation_df, scatter_title + " (Moving Average 25%)", show_moving_average=True)

    c.drawImage(chart_path_prey_vs_predator, x_positions[0], y, width=graph_width, height=graph_height)
    os.remove(chart_path_prey_vs_predator)

    c.drawImage(chart_path_prey_vs_predator_moving_avg, x_positions[1], y, width=graph_width, height=graph_height)
    os.remove(chart_path_prey_vs_predator_moving_avg)

    # Gráficos adicionais (histograma e boxplot) abaixo dos gráficos principais
    y -= vertical_gap
    # Histograma de População e Box Plot lado a lado
    chart_path_histogram = generate_population_histogram(simulation_df)
    chart_path_boxplot = generate_population_boxplot(simulation_df)

    c.drawImage(chart_path_histogram, x_positions[0], y, width=graph_width, height=graph_height)
    c.drawImage(chart_path_boxplot, x_positions[1], y, width=graph_width, height=graph_height)
    os.remove(chart_path_histogram)
    os.remove(chart_path_boxplot)

    return y - graph_height - 20


def generate_population_episode_chart(simulation_df, y_column, color, title, show_moving_average=False, sma_percentage=0.25):
    plt.figure(figsize=(6, 4))
    
    if show_moving_average:
        window_size = max(1, int(len(simulation_df) * sma_percentage))
        data_smoothed = simulation_df[y_column].rolling(window=window_size, center=True).mean()
        plt.plot(simulation_df['Episode'], data_smoothed, color=color, label=f'{y_column} Moving Average')
    else:
        plt.plot(simulation_df['Episode'], simulation_df[y_column], color=color, label=f'{y_column} Count')
    
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(f"{y_column} Count")
    plt.legend()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        chart_path = tmpfile.name
        plt.savefig(chart_path, format='png')
    plt.close()
    
    return chart_path

def generate_comparison_chart(simulation_df, title, show_moving_average=False, sma_percentage=0.25):
    plt.figure(figsize=(6, 4))
    
    if show_moving_average:
        window_size = max(1, int(len(simulation_df) * sma_percentage))
        prey_smoothed = simulation_df['Prey'].rolling(window=window_size, center=True).mean()
        predator_smoothed = simulation_df['Predator'].rolling(window=window_size, center=True).mean()
        plt.plot(simulation_df['Episode'], prey_smoothed, color='blue', linestyle='--', label='Prey Moving Average')
        plt.plot(simulation_df['Episode'], predator_smoothed, color='red', linestyle='--', label='Predator Moving Average')
    else:
        plt.plot(simulation_df['Episode'], simulation_df['Prey'], color='blue', label='Prey Count')
        plt.plot(simulation_df['Episode'], simulation_df['Predator'], color='red', label='Predator Count')
    
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.legend()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        chart_path = tmpfile.name
        plt.savefig(chart_path, format='png')
    plt.close()
    
    return chart_path

def generate_prey_vs_predator_chart(simulation_df, title, show_moving_average=False, sma_percentage=0.25):
    plt.figure(figsize=(6, 4))
    
    if show_moving_average:
        window_size = max(1, int(len(simulation_df) * sma_percentage))
        prey_smoothed = simulation_df['Prey'].rolling(window=window_size, center=True).mean()
        predator_smoothed = simulation_df['Predator'].rolling(window=window_size, center=True).mean()
        plt.plot(prey_smoothed, predator_smoothed, color='green', linestyle='--', label='Moving Average')
    else:
        plt.scatter(simulation_df['Prey'], simulation_df['Predator'], color='blue', label='Data Points')
        m, b = np.polyfit(simulation_df['Prey'], simulation_df['Predator'], 1)
        plt.plot(simulation_df['Prey'], m * simulation_df['Prey'] + b, color='red', label='Linear Regression')
    
    plt.title(title)
    plt.xlabel("Prey Population")
    plt.ylabel("Predator Population")
    plt.legend()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        chart_path = tmpfile.name
        plt.savefig(chart_path, format='png')
    plt.close()
    
    return chart_path

def generate_regression_chart(simulation_df, title, show_moving_average=False, sma_percentage=0.25):
    plt.figure(figsize=(6, 4))
    
    if show_moving_average:
        window_size = max(1, int(len(simulation_df) * sma_percentage))
        prey_smoothed = simulation_df['Prey'].rolling(window=window_size, center=True).mean()
        predator_smoothed = simulation_df['Predator'].rolling(window=window_size, center=True).mean()
        plt.plot(prey_smoothed, predator_smoothed, color='green', linestyle='--', label=f'Moving Average Trajectory ({window_size} points)')
    else:
        plt.scatter(simulation_df['Prey'], simulation_df['Predator'], color='blue', label='Data Points')
        m, b = np.polyfit(simulation_df['Prey'], simulation_df['Predator'], 1)
        plt.plot(simulation_df['Prey'], m * simulation_df['Prey'] + b, color='red', label='Linear Regression')

    plt.title(title)
    plt.xlabel("Prey Count")
    plt.ylabel("Predator Count")
    plt.legend()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        chart_path = tmpfile.name
        plt.savefig(chart_path, format='png')
    plt.close()

    return chart_path

def generate_population_histogram(simulation_df):
    # Gera o histograma para `Prey` e `Predator`
    plt.figure(figsize=(6, 4))
    plt.hist(simulation_df['Prey'], bins=15, color='blue', alpha=0.5, label='Prey Population')
    plt.hist(simulation_df['Predator'], bins=15, color='red', alpha=0.5, label='Predator Population')
    
    plt.title("Population Histogram of Prey and Predator")
    plt.xlabel("Population Count")
    plt.ylabel("Frequency")
    plt.legend()

    # Salva o gráfico como uma imagem temporária
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        chart_path = tmpfile.name
        plt.savefig(chart_path, format='png')
    plt.close()

    return chart_path

def generate_population_boxplot(simulation_df):
    # Gera o boxplot para `Prey` e `Predator`
    plt.figure(figsize=(6, 4))
    plt.boxplot([simulation_df['Prey'], simulation_df['Predator']], labels=['Prey', 'Predator'])
    
    plt.title("Box Plot of Prey and Predator Populations")
    plt.ylabel("Population Count")

    # Salva o gráfico como uma imagem temporária
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        chart_path = tmpfile.name
        plt.savefig(chart_path, format='png')
    plt.close()

    return chart_path

def on_generate_pdf_button_click_old():
    # Mostra caixa de diálogo para selecionar o arquivo JSON
    json_file_path = os.path.join('save', 'models', 'mi_15.json')
    if not os.path.exists(json_file_path):
        messagebox.showerror("Error", f"File not found: {json_file_path}")
        return

    data = load_json_data(json_file_path)
    
    if data:
        model_name = data.get('model', 'Unknown Model')
        creation_date = datetime.now().strftime("%Y-%m-%d")

        output_pdf_path = f"save/reports/prime_report_{model_name}_{creation_date}.pdf"
        generate_pdf_report(data, output_pdf_path)

def on_generate_pdf_button_click(input_json_file):
    # Mostra caixa de diálogo para selecionar o arquivo JSON
    json_file_path = os.path.join('save', 'models', input_json_file)
    if not os.path.exists(json_file_path):
        messagebox.showerror("Error", f"File not found: {json_file_path}")
        return

    data = load_json_data(json_file_path)
    
    if data:
        model_name = data.get('model', 'Unknown Model')
        creation_date = datetime.now().strftime("%Y-%m-%d")

        output_pdf_path = f"save/reports/prime_report_{model_name}_{creation_date}.pdf"
        generate_pdf_report(data, output_pdf_path)
