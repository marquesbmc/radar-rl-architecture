import sys
import os
import json
import tkinter as tk
import threading
import time
import re
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd


from scipy.stats import iqr

from tkinter import ttk, filedialog, messagebox
from ttkthemes import ThemedTk  # Importando ttkthemes para maior flexibilidade
from core.Agents import Prey, Predator
from core.Env import Env
from core.Sim import Sim
from neuralnetwork.architecture import nn_dqn, nn_per, nn_dueling, nn_double, nn_radar_dqn, nn_radar_per, nn_radar_dueling, nn_radar_double
from monitor import preparar_e_chamar_visualizacao
from report import on_generate_pdf_button_click

# Adiciona o diretório 'core' ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'core')))

selected_file_name = None # Variável global para armazenar o nome do arquivo selecionado
simulation_running = False  # Variável para controlar o estado da simulação
# Variável global para armazenar os dados de simulação carregados
global_simulation_data = None  # Inicialmente, a variável está vazia

# Lista de imagens a verificar
image_files = [
    "analysis_image.png",
    "cancel_button.png",
    "clear_button.png",
    "report_button.png",
    "exit_button.png",
    "logo.png",
    "title.png",
    "modeling_image.png",
    "open_button.png",
    "run_button.png",
    "save_button.png",
    "simulation_image.png",
    "start_button.png"
]


# Função para carregar imagens com segurança
def load_image(image_name):
    try:
        image_directory = os.path.join(os.path.dirname(__file__), 'images')  # Diretório das imagens
        return tk.PhotoImage(file=os.path.join(image_directory, image_name)).subsample(1, 1)
    except Exception as e:
        print(f"Erro ao carregar a imagem {image_name}: {e}")
        return None  # Retornar None se não conseguir carregar a imagem


def validate_simulation_labels():
    # Verifica se todos os valores importantes estão preenchidos
    if not sim_name_value.cget("text"):
        messagebox.showerror("Error", "Model Name is required.")
        return False
    if not sim_grids_value.cget("text"):
        messagebox.showerror("Error", "Grid Size is required.")
        return False
    if not sim_episodes_value.cget("text"):
        messagebox.showerror("Error", "Episodes is required.")
        return False
    if not sim_steps_value.cget("text"):
        messagebox.showerror("Error", "Steps is required.")
        return False
    if not sim_prey_inicial_value.cget("text"):
        messagebox.showerror("Error", "Initial Prey Count is required.")
        return False
    if not sim_predator_inicial_value.cget("text"):
        messagebox.showerror("Error", "Initial Predator Count is required.")
        return False
    if not sim_prey_limite_value.cget("text"):
        messagebox.showerror("Error", "Max Prey Count is required.")
        return False
    if not sim_predator_limite_value.cget("text"):
        messagebox.showerror("Error", "Max Predator Count is required.")
        return False
    if not sim_prey_reproduction_value.cget("text"):
        messagebox.showerror("Error", "Prey Spawn Rate is required.")
        return False
    if not sim_predator_reproduction_value.cget("text"):
        messagebox.showerror("Error", "Predator Spawn Rate is required.")
        return False
    if not sim_prey_fatigue_value.cget("text"):
        messagebox.showerror("Error", "Prey Step Decay is required.")
        return False
    if not sim_predator_fatigue_value.cget("text"):
        messagebox.showerror("Error", "Predator Step Decay is required.")
        return False
    if not sim_prey_learning_value.cget("text"):
        messagebox.showerror("Error", "Prey Learning Model is required.")
        return False
    if not sim_predator_learning_value.cget("text"):
        messagebox.showerror("Error", "Predator Learning Model is required.")
        return False
    if not sim_prey_design_value.cget("text"):
        messagebox.showerror("Error", "Prey Advanced Layer is required.")
        return False
    if not sim_predator_design_value.cget("text"):
        messagebox.showerror("Error", "Predator Advanced Layer is required.")
        return False
    
    # Se todos os valores estiverem preenchidos corretamente
    return True


def check_images_with_delay(delay_ms=1):
    image_directory = os.path.join(os.path.dirname(__file__), 'images')  # Diretório das imagens
    
    missing_images = []
    
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        
        if not os.path.exists(image_path):
            missing_images.append(image_file)
        
        # Adicionar um delay de 500ms (ou valor customizado)
        time.sleep(delay_ms / 1000)  # Converte milissegundos para segundos

    if missing_images:
        messagebox.showerror("Error", f"Missing image files: {', '.join(missing_images)}")
        return False
    
    return True


class RedirectedConsole:
    def __init__(self, text_widget, scrollbar):
        self.text_widget = text_widget
        self.scrollbar = scrollbar
        self.original_stdout = sys.__stdout__  # Salva o stdout original

    def write(self, string):
        """Redireciona a saída para o widget de texto com liberação imediata."""
        # Verifica se há algo a escrever
        if len(string) > 0:
            current_scroll_position = self.text_widget.yview()

            # Inserir o novo texto no final do widget
            self.text_widget.config(state=tk.NORMAL)  # Habilita edição temporária
            self.text_widget.insert(tk.END, string)
            self.text_widget.config(state=tk.DISABLED)  # Desabilita para evitar edição do usuário
            
            # Atualiza o widget gráfico apenas se necessário
            if current_scroll_position[1] >= 0.98:
                self.text_widget.yview(tk.END)  # Scroll automático
                self.text_widget.update_idletasks()  # Força a atualização da interface gráfica

    def flush(self):
        """Libera o buffer de saída (necessário para garantir compatibilidade com print)."""
        pass


def validate_inputs():
    model_name = model_name_entry.get()
    if not model_name or len(model_name) < 3 or len(model_name) > 14 or re.search(r'\s', model_name):
        messagebox.showerror("Error", "Model name must be between 3 and 14 alphanumeric characters without spaces.")
        return False
    
    grids = grids_entry.get()
    if not grids.isdigit() or int(grids) <= 0:
        messagebox.showerror("Error", "Grid Size must be a numeric value greater than 0.")
        return False

    episodes = episodes_entry.get()
    if not episodes.isdigit() or int(episodes) <= 0:
        messagebox.showerror("Error", "Episodes must be a numeric value greater than 0.")
        return False

    steps = steps_entry.get()
    if not steps.isdigit() or int(steps) < 0:
        messagebox.showerror("Error", "Steps must be a numeric value greater than 0.")
        return False

    prey_initial = prey_inicial_entry.get()
    if not prey_initial.isdigit() or int(prey_initial) < 0:
        messagebox.showerror("Error", "Prey Initial Count must be a numeric value greater than 0.")
        return False

    predator_initial = predator_inicial_entry.get()
    if not predator_initial.isdigit() or int(predator_initial) <= 0:
        messagebox.showerror("Error", "Predator Initial Count must be a numeric value greater than 0.")
        return False

    prey_limit = prey_limite_entry.get()
    if not prey_limit.isdigit() or None:
        messagebox.showerror("Error", "Prey Max Count must be a numeric value greater than or equal to 0 or null.")
        return False

    predator_limit = predator_limite_entry.get()
    if not predator_limit.isdigit() or None:
        messagebox.showerror("Error", "Predator Max Count must be a numeric value greater than or equal to 0 or null.")
        return False

    prey_reproduction = prey_reproduction_entry.get()
    if not prey_reproduction.isdigit() or int(prey_reproduction) < 0 or int(prey_reproduction) > 100:
        messagebox.showerror("Error", "Prey Spawn Rate must be a numeric value between 0% and 100%.")
        return False

    predator_reproduction = predator_reproduction_entry.get()
    if not predator_reproduction.isdigit() or int(predator_reproduction) < 0 or int(predator_reproduction) > 100:
        messagebox.showerror("Error", "Predator Spawn Rate must be a numeric value between 0% and 100%.")
        return False

    prey_fatigue = prey_fatigue_entry.get()
    if not prey_fatigue.isdigit() or int(prey_fatigue) <= 0:
        messagebox.showerror("Error", "Prey Step Decay must be a numeric value greater than 0.")
        return False

    predator_fatigue = predator_fatigue_entry.get()
    if not predator_fatigue.isdigit() or int(predator_fatigue) <= 0:
        messagebox.showerror("Error", "Predator Step Decay must be a numeric value greater than 0.")
        return False

    # Validação para os modelos de aprendizado
    prey_learning_model = prey_learning_var.get()
    if not prey_learning_model:
        messagebox.showerror("Error", "Please select a learning model for Prey.")
        return False

    predator_learning_model = predator_learning_var.get()
    if not predator_learning_model:
        messagebox.showerror("Error", "Please select a learning model for Predator.")
        return False

    # Validação para as variantes de design
    prey_design_variant = prey_design_var.get()
    if not prey_design_variant:
        messagebox.showerror("Error", "Please select a advanced layer for Prey.")
        return False

    predator_design_variant = predator_design_var.get()
    if not predator_design_variant:
        messagebox.showerror("Error", "Please select a advanced layer for Predator.")
        return False


    return True


def on_slider_change(value):
    slider_label.config(text=f"Grid Size: {int(float(value))}")


def read_simulation():
    try:
        # Definir o caminho inicial como 'save/simulations' com relação ao diretório do arquivo executado
        script_directory = os.path.dirname(os.path.abspath(__file__))  # Diretório do arquivo .py
        initial_directory = os.path.join(script_directory, 'save', 'models')  # Caminho completo para 'save/models'

        # Verificar se a pasta 'save/simulations' existe, caso contrário, mostrar erro
        if not os.path.exists(initial_directory):
            messagebox.showerror("Error", f"Directory not found: {initial_directory}")
            return

        # Abrir a caixa de diálogo no diretório 'save/simulations'
        file_path = filedialog.askopenfilename(
            initialdir=initial_directory,  # Definir a pasta inicial
            title="Open JSON File",
            filetypes=(("JSON Files", "*.json"), ("All Files", "*.*"))
        )

        if not file_path:
            return

        # Abrir o arquivo JSON e carregar os dados
        with open(file_path, "r") as json_file:
            data = json.load(json_file)

        # Atualizar os labels usando o método config()
        sim_name_value.config(text=data.get("model", ""))

        sim_grids_value.config(text=data.get("environment", {}).get("grid_size", ""))
        sim_episodes_value.config(text=data.get("environment", {}).get("episodes", ""))
        sim_steps_value.config(text=data.get("environment", {}).get("steps", ""))

        sim_prey_inicial_value.config(text=data.get("population", {}).get("prey_initial_count", ""))
        sim_predator_inicial_value.config(text=data.get("population", {}).get("predator_initial_count", ""))

        sim_prey_limite_value.config(text=data.get("population", {}).get("prey_max_count", ""))
        sim_predator_limite_value.config(text=data.get("population", {}).get("predator_max_count", ""))

        sim_prey_reproduction_value.config(text=data.get("population", {}).get("prey_spawn_rate", ""))
        sim_predator_reproduction_value.config(text=data.get("population", {}).get("predator_spawn_rate", ""))

        sim_prey_fatigue_value.config(text=data.get("population", {}).get("prey_step_decay", ""))
        sim_predator_fatigue_value.config(text=data.get("population", {}).get("predator_step_decay", ""))

        sim_prey_learning_value.config(text=data.get("neural_network", {}).get("prey_learning_model", ""))
        sim_predator_learning_value.config(text=data.get("neural_network", {}).get("predator_learning_model", ""))

        sim_prey_design_value.config(text=data.get("neural_network", {}).get("prey_design_variant", ""))
        sim_predator_design_value.config(text=data.get("neural_network", {}).get("predator_design_variant", ""))

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load simulation data: {str(e)}")


def read_review_data(data):
    try:

        # Atualizar os labels usando o método config()
        rev_name_value.config(text=data.get("model", ""))

        rev_grids_value.config(text=data.get("environment", {}).get("grid_size", ""))
        rev_episodes_value.config(text=data.get("environment", {}).get("episodes", ""))
        rev_steps_value.config(text=data.get("environment", {}).get("steps", ""))

        rev_prey_inicial_value.config(text=data.get("population", {}).get("prey_initial_count", ""))
        rev_predator_inicial_value.config(text=data.get("population", {}).get("predator_initial_count", ""))

        rev_prey_limite_value.config(text=data.get("population", {}).get("prey_max_count", ""))
        rev_predator_limite_value.config(text=data.get("population", {}).get("predator_max_count", ""))

        rev_prey_reproduction_value.config(text=data.get("population", {}).get("prey_spawn_rate", ""))
        rev_predator_reproduction_value.config(text=data.get("population", {}).get("predator_spawn_rate", ""))

        rev_prey_fatigue_value.config(text=data.get("population", {}).get("prey_step_decay", ""))
        rev_predator_fatigue_value.config(text=data.get("population", {}).get("predator_step_decay", ""))

        rev_prey_learning_value.config(text=data.get("neural_network", {}).get("prey_learning_model", ""))
        rev_predator_learning_value.config(text=data.get("neural_network", {}).get("predator_learning_model", ""))

        rev_prey_design_value.config(text=data.get("neural_network", {}).get("prey_design_variant", ""))
        rev_predator_design_value.config(text=data.get("neural_network", {}).get("predator_design_variant", ""))

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load simulation data: {str(e)}")


def read_quantitative_data(data):
    # Verificar se `quantitative_data` existe em `data`
    quantitative_data = data.get("quantitative_data", None)
    if not quantitative_data:
        print("Error: 'quantitative_data' not found in the JSON data.")
        return None

    # Extrair os dados de `predator` e `prey`
    predator_data = quantitative_data.get("predator", [])
    prey_data = quantitative_data.get("prey", [])

    # Converter os dados de `predator` e `prey` para DataFrames separados
    predator_df = pd.DataFrame(predator_data, columns=["Reward", "Done", "Step"])
    prey_df = pd.DataFrame(prey_data, columns=["Reward", "Done", "Step"])

    # Adicionar uma coluna para diferenciar `predator` e `prey`
    predator_df["Type"] = "Predator"
    prey_df["Type"] = "Prey"

    # Concatenar os dois DataFrames em um só
    quantitative_df = pd.concat([predator_df, prey_df], ignore_index=True)

    return quantitative_df


def read_behavior_data(data):
    """
    Lê os dados de `behavior_data` a partir do JSON e converte-os em um DataFrame com as colunas 
    organizadas na ordem: Type, Behavior, Count.
    
    Args:
        data (dict): Dicionário que contém o JSON dos dados.
    
    Returns:
        pd.DataFrame: DataFrame com os dados de comportamento (`behavior_data`) para predadores e presas.
    """
    # Verificar se `behavior_data` existe em `data`
    behavior_data = data.get("behavior_data", None)
    if not behavior_data:
        print("Error: 'behavior_data' not found in the JSON data.")
        return None

    # Extrair os dados de `predator` e `prey`
    predator_behavior = behavior_data.get("predator", {})
    prey_behavior = behavior_data.get("prey", {})

    # Converter os dados de `predator` e `prey` para DataFrames separados
    predator_df = pd.DataFrame(list(predator_behavior.items()), columns=["Behavior", "Count"])
    prey_df = pd.DataFrame(list(prey_behavior.items()), columns=["Behavior", "Count"])

    # Adicionar uma coluna para diferenciar `predator` e `prey`
    predator_df["Type"] = "Predator"
    prey_df["Type"] = "Prey"

    # Concatenar os dois DataFrames em um só e reorganizar as colunas na ordem desejada
    behavior_df = pd.concat([predator_df, prey_df], ignore_index=True)
    behavior_df = behavior_df[["Type", "Behavior", "Count"]]  # Reordenar colunas

    return behavior_df


def read_simulation_data(data):
    try:
         # Capturar os dados do campo 'simulation_data' e armazenar no DataFrame
        simulation_data = data.get("simulation_data", [])
        if not simulation_data:
            messagebox.showwarning("Warning", "No simulation data found.")
            return None

        # Criar DataFrame a partir da simulation_data
        df = pd.DataFrame(simulation_data, columns=['episode', 'prey', 'predator'])
        return df

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load simulation data: {str(e)}")
        return None


def open_data_for_model():
    try:
        # Definir o caminho inicial como 'save/models' com relação ao diretório do arquivo executado
        script_directory = os.path.dirname(os.path.abspath(__file__))  # Diretório do arquivo .py
        initial_directory = os.path.join(script_directory, 'save', 'models')  # Caminho completo para 'save/models'

        # Verificar se a pasta 'save/models' existe, caso contrário, mostrar erro
        if not os.path.exists(initial_directory):
            messagebox.showerror("Error", f"Directory not found: {initial_directory}")
            return

        # Abrir a caixa de diálogo no diretório 'save/models'
        file_path = filedialog.askopenfilename(
            initialdir=initial_directory,  # Definir a pasta inicial
            title="Open JSON File",
            filetypes=(("JSON Files", "*.json"), ("All Files", "*.*"))
        )
        
        if not file_path:
            return

        # Abrir o arquivo JSON e carregar os dados
        with open(file_path, "r") as json_file:
            data = json.load(json_file)

        # Limpar e preencher os campos com os dados carregados
        model_name_entry.delete(0, tk.END)
        model_name_entry.insert(0, data.get("model", ""))

        grids_entry.delete(0, tk.END)
        grids_entry.insert(0, data.get("environment", {}).get("grid_size", ""))

        episodes_entry.delete(0, tk.END)
        episodes_entry.insert(0, data.get("environment", {}).get("episodes", ""))

        steps_entry.delete(0, tk.END)
        steps_entry.insert(0, data.get("environment", {}).get("steps", ""))

        prey_inicial_entry.delete(0, tk.END)
        prey_inicial_entry.insert(0, data.get("population", {}).get("prey_initial_count", ""))

        predator_inicial_entry.delete(0, tk.END)
        predator_inicial_entry.insert(0, data.get("population", {}).get("predator_initial_count", ""))

        prey_limite_entry.delete(0, tk.END)
        prey_limite_entry.insert(0, data.get("population", {}).get("prey_max_count", ""))

        predator_limite_entry.delete(0, tk.END)
        predator_limite_entry.insert(0, data.get("population", {}).get("predator_max_count", ""))

        prey_reproduction_entry.delete(0, tk.END)
        prey_reproduction_entry.insert(0, data.get("population", {}).get("prey_spawn_rate", ""))

        predator_reproduction_entry.delete(0, tk.END)
        predator_reproduction_entry.insert(0, data.get("population", {}).get("predator_spawn_rate", ""))

        prey_fatigue_entry.delete(0, tk.END)
        prey_fatigue_entry.insert(0, data.get("population", {}).get("prey_step_decay", ""))

        predator_fatigue_entry.delete(0, tk.END)
        predator_fatigue_entry.insert(0, data.get("population", {}).get("predator_step_decay", ""))

        prey_learning_var.set(data.get("neural_network", {}).get("prey_learning_model", ""))
        predator_learning_var.set(data.get("neural_network", {}).get("predator_learning_model", ""))

        prey_design_var.set(data.get("neural_network", {}).get("prey_design_variant", ""))
        predator_design_var.set(data.get("neural_network", {}).get("predator_design_variant", ""))


    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {str(e)}")

        # Atualizando o valor de selected_file_name com o valor do arquivo JSON
        global selected_file_name
        selected_file_name = data.get("Selected File", "")

        # Selecionar automaticamente o arquivo correto no listbox
        listbox_items = listbox_neural_network.get(0, tk.END)  # Pegar todos os itens do listbox
        for index, item in enumerate(listbox_items):
            if selected_file_name in item:
                listbox_neural_network.select_set(index)  # Selecionar o arquivo correspondente
                listbox_neural_network.see(index)  # Garantir que o item fique visível
                break

        messagebox.showinfo("Success", "Data loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {str(e)}")


def run_simulate_in_thread(console_output=None):
    global simulation_running
    global run_button_simulate


    if simulation_running:
        print("Simulation is already running.")
        return
    
    # Adicionar validação antes de iniciar a simulação
    if not validate_simulation_labels():
        return  # Interrompe a execução se a validação falhar

    # Definir simulation_running como True antes de iniciar a thread
    simulation_running = True
    print("Starting simulation in a new thread...")

    def run_simulation():
        try:
            # Capturar os valores da aba Simulation antes de iniciar a simulação
            print("Capturing simulation parameters...")
            model_name = sim_name_value.cget("text")  # Pega o valor atual do label
            grid_size = int(sim_grids_value.cget("text"))  # Pega o valor atual do label
            episodes = int(sim_episodes_value.cget("text"))  # Pega o valor atual do label
            steps = int(sim_steps_value.cget("text"))  # Pega o valor atual do label

            # Capturar os dados de Prey (presas)
            prey_initial_count = int(sim_prey_inicial_value.cget("text"))  # Contagem inicial de presas
            prey_max_count = int(sim_prey_limite_value.cget("text"))  # Contagem máxima de presas
            prey_spawn_rate = int(sim_prey_reproduction_value.cget("text"))  # Taxa de reprodução das presas
            prey_step_decay = int(sim_prey_fatigue_value.cget("text"))  # Fadiga das presas (decaimento por passo)
            prey_learning_model = sim_prey_learning_value.cget("text")  # Modelo de aprendizado das presas
            prey_design_variant = sim_prey_design_value.cget("text")  # Variante de design das presas

            
            # Capturar os dados de Predator (predadores)
            predator_initial_count = int(sim_predator_inicial_value.cget("text"))  # Contagem inicial de predadores
            predator_max_count = int(sim_predator_limite_value.cget("text"))  # Contagem máxima de predadores
            predator_spawn_rate = int(sim_predator_reproduction_value.cget("text"))  # Taxa de reprodução dos predadores
            predator_step_decay = int(sim_predator_fatigue_value.cget("text"))  # Fadiga dos predadores (decaimento por passo)
            predator_learning_model = sim_predator_learning_value.cget("text")  # Modelo de aprendizado dos predadores
            predator_design_variant = sim_predator_design_value.cget("text")  # Variante de design dos predadores

            run_simulate(console_output,
                          model_name, grid_size, episodes, steps,
                          prey_initial_count, prey_max_count, prey_spawn_rate, prey_step_decay, prey_learning_model, prey_design_variant, 
                          predator_initial_count, predator_max_count, predator_spawn_rate, predator_step_decay, predator_learning_model, predator_design_variant
                         )  # A função de simulação é chamada aqui

        finally:
            # Após a simulação, garantir que o botão volte para 'Run'
            root.after(0, toggle_run_cancel)

    # Criar uma nova thread para a simulação
    simulation_thread = threading.Thread(target=run_simulation)
    simulation_thread.daemon = True  # Isso garante que a thread seja fechada com a aplicação
    simulation_thread.start()

    # Trocar o botão para 'Cancel' imediatamente
    root.after(0, toggle_run_cancel)


def run_simulate(console_output, model_name, grid_size, episodes, steps, prey_initial_count, 
                 prey_max_count, prey_spawn_rate, prey_step_decay, prey_learning_model, prey_design_variant,
                 predator_initial_count, predator_max_count, predator_spawn_rate, predator_step_decay, predator_learning_model, predator_design_variant
                 ):
    global simulation_running  # Referenciar a variável global

    # Redireciona stdout para o console Tkinter na thread secundária
    sys.stdout = RedirectedConsole(console_output, None)
    sys.stderr = sys.stdout  # Redirecionar stderr também, se necessário


    def tk_print(*args, **kwargs):
        message = ' '.join(map(str, args)) + '\n'
        if console_output:
            # Captura a posição atual do scroll
            current_scroll_position = console_output.yview()

            # Inserir a nova mensagem no console
            console_output.config(state=tk.NORMAL)  # Habilitar edição temporária
            console_output.insert(tk.END, message)  # Inserir a nova mensagem

            # Verificar se o scroll está no final (se o valor de current_scroll_position[1] é 1.0)
            if current_scroll_position[1] == 1.0:
                console_output.see(tk.END)  # Rolar para o final automaticamente se já estiver no final
            # Caso contrário, não mover o scroll (deixa o usuário na posição manual que ele selecionou)

            # Atualizar a interface gráfica
            console_output.update_idletasks()
            console_output.config(state=tk.DISABLED)  # Desativar para evitar edição do usuário
        else:
            print(*args, **kwargs)  # Se não houver console_output, imprime no console padrão


    simulation_running = True  # Marcar como em execução

    try:
        tk_print("Starting the simulation process for the model...")
        # Configurações de ambiente e modelo
        task_name = model_name
        size = grid_size
        # directory = "/neuralnetwork"
        num_prey = prey_initial_count
        num_predator = predator_initial_count
        num_episodes = episodes
        num_steps = steps
        input_shape = (7, 7, 3)

        # deleta sim_
        directory = os.path.join("save", "sim")
        filename = f"sim_{model_name}.csv"
        delete_file(directory, filename)

        # Diretório relativo ao script atual
        directory_weights = os.path.join(os.path.dirname(__file__), 'neuralnetwork/weights')

        env = Env(sizeX=size, sizeY=size, ray=3)

        
        print(f"Model Neural: Prey = {prey_learning_model.lower()}, Predator = {predator_learning_model.lower()}")
        print(f"Design Variant: Prey = {prey_design_variant.lower()}, Predator = {predator_design_variant.lower()}")


        if "none" in prey_design_variant.lower():
            prey_name = f"{prey_learning_model.lower()}"
        else:
            prey_name = f"{prey_design_variant.lower()}_{prey_learning_model.lower()}"

        if "none" in predator_design_variant.lower():
            predator_name = f"{predator_learning_model.lower()}"
        else:
            predator_name = f"{predator_design_variant.lower()}_{predator_learning_model.lower()}"
        
        print("Please wait... Processing your request.")



        # Inicializa e configura a rede neural da presa
        if "double" in prey_learning_model.lower():
            
            model_nn_prey = globals()[f"nn_{prey_name}"]()
            dummy_input = tf.random.uniform((1, *input_shape))
            model_nn_prey(dummy_input)  # Ativa a rede principal
            model_nn_prey.target_call(dummy_input)  # Ativa a rede-alvo
            model_nn_prey.update_target_network()  # Cria e sincroniza a rede-alvo
            model_nn_prey.load_weights(os.path.join(directory_weights, f"weights_prey_{type(model_nn_prey).__name__}.h5"))
            
        else:
            model_nn_prey = globals()[f"nn_{prey_name}"]()
            model_nn_prey.build(input_shape=(None,) + input_shape)
            model_nn_prey.load_weights(os.path.join(directory_weights, f"weights_prey_{type(model_nn_prey).__name__}.h5"))



        if "double" in predator_learning_model.lower():

            model_nn_predator = globals()[f"nn_{predator_name}"]()
            dummy_input = tf.random.uniform((1, *input_shape))
            model_nn_predator(dummy_input)  # Ativa a rede principal
            model_nn_predator.target_call(dummy_input)  # Ativa a rede-alvo
            model_nn_predator.update_target_network()  # Cria e sincroniza a rede-alvo
            model_nn_predator.load_weights(os.path.join(directory_weights, f"weights_predator_{type(model_nn_predator).__name__}.h5"))

        else:
            model_nn_predator = globals()[f"nn_{predator_name.lower()}"]()
            model_nn_predator.build(input_shape=(None,) + input_shape)
            model_nn_predator.load_weights(os.path.join(directory_weights, f"weights_predator_{type(model_nn_predator).__name__}.h5"))

        for j in range(num_prey):
            if not simulation_running:
                tk_print("Simulation cancelled.")
                break
            pos = env.new_position()
            if pos:
                x, y = pos
                prey = Prey(x, y, env, id=j, fatigue_max=prey_step_decay, percent_rep=prey_spawn_rate, nn=model_nn_prey)
                env.add_agent(prey)
                tk_print(f"Added Prey agent {j} at position ({x}, {y}).")

        for i in range(num_predator):
            if not simulation_running:
                tk_print("Simulation cancelled.")
                break
            pos = env.new_position()
            if pos:
                x, y = pos
                predator = Predator(x, y, env, id=i, fatigue_max=predator_step_decay, percent_rep=predator_spawn_rate, nn=model_nn_predator)
                env.add_agent(predator)
                tk_print(f"Added Predator agent {i} at position ({x}, {y}).")

        tk_print("Environment setup complete.")

        tk_print("=== initialized Simulation ===")

        # Criar a sessão de teste
        simulation = Sim(task_name, env, num_predator, num_prey, predator_max_count, prey_max_count, model_nn_predator, model_nn_prey, num_episodes=num_episodes, num_steps=num_steps)
                 
        # Simulação de exemplo (inserir lógica real aqui)
        for result in simulation.run():
            if not simulation_running:
                break
            tk_print(result)


        #on_run_complete(sim_name_value)

        # Verificar se a simulação não foi cancelada antes de imprimir a mensagem de conclusão
        if simulation_running:
            tk_print("=== Simulation Completed ===")
        else:
            tk_print("=== Simulation cancelled ===")

    except Exception as e:
        tk_print(f" === An error occurred during simulation: {e}  ===")
        raise

    finally:
        simulation_running = False  # Marcar como finalizada


def merge_simulation_data(latest_file_path, model_file_path):
    # Carrega os dados do arquivo latest_file
    with open(latest_file_path, 'r') as f:
        latest_data = json.load(f)
    
    # Carrega os dados do arquivo modelo (teste_sim.json)
    with open(model_file_path, 'r') as f:
        model_data = json.load(f)

    # Adiciona os dados de latest_file ao campo "simulation_data" de model_data
    if "simulation_data" in latest_data and isinstance(latest_data["simulation_data"], list):
        model_data["simulation_data"].extend(latest_data["simulation_data"])
    else:
        print("Nenhum dado de simulação encontrado em latest_file.")

    # Salva o arquivo modelo atualizado
    with open(model_file_path, 'w') as f:
        json.dump(model_data, f, indent=4)
    print("Dados de simulação mesclados com sucesso no arquivo:", model_file_path)


def toggle_run_cancel():
    global run_button_simulation
    global cancel_button_simulation
    global simulation_running

    if simulation_running:
        run_button_simulation.grid_remove()  # Remove o botão 'Run'
        cancel_button_simulation.grid(row=0, column=3, padx=10, pady=10, sticky="e")  # Adiciona o botão 'Cancel'
    else:
        cancel_button_simulation.grid_remove()  # Remove o botão 'Cancel'
        run_button_simulation.grid(row=0, column=3, padx=10, pady=10, sticky="e")  # Adiciona o botão 'Run'


def cancel_simulation():
    global simulation_running
    if simulation_running:
        simulation_running = False  # Marcar como cancelada
        print("Cancelling the simulation...")
        toggle_run_cancel()  # Reverter o botão para "Run" após o cancelamento


def clear_fields():


    # Limpar campos de texto
    model_name_entry.delete(0, tk.END)
    grids_entry.delete(0, tk.END)
    episodes_entry.delete(0, tk.END)
    steps_entry.delete(0, tk.END)
    
    prey_inicial_entry.delete(0, tk.END)
    predator_inicial_entry.delete(0, tk.END)
    prey_limite_entry.delete(0, tk.END)
    predator_limite_entry.delete(0, tk.END)
    
    prey_reproduction_entry.delete(0, tk.END)
    predator_reproduction_entry.delete(0, tk.END)
    prey_fatigue_entry.delete(0, tk.END)
    predator_fatigue_entry.delete(0, tk.END)

    # Limpar comboboxes (neural network )
    prey_learning_menu.set('')  # Limpar o valor selecionado no combobox
    predator_learning_menu.set('')
    
    prey_design_menu.set('')
    predator_design_menu.set('')


    # Exibir mensagem de confirmação
    print("Todos os campos foram limpos com sucesso.")


def clear_fields_sim():
    # Limpar labels de valores da aba Simulation
    sim_name_value.config(text="")
    sim_grids_value.config(text="")
    sim_episodes_value.config(text="")
    sim_steps_value.config(text="")
    
    sim_prey_inicial_value.config(text="")
    sim_predator_inicial_value.config(text="")
    sim_prey_limite_value.config(text="")
    sim_predator_limite_value.config(text="")
    
    sim_prey_reproduction_value.config(text="")
    sim_predator_reproduction_value.config(text="")
    sim_prey_fatigue_value.config(text="")
    sim_predator_fatigue_value.config(text="")
    
    sim_prey_learning_value.config(text="")
    sim_predator_learning_value.config(text="")
    sim_prey_design_value.config(text="")
    sim_predator_design_value.config(text="")
    sim_prey_attention_value.config(text="")
    sim_predator_attention_value.config(text="")
    
    # Limpar o console (widget Text)
    console_text.config(state=tk.NORMAL)  # Habilitar edição no console
    console_text.delete('1.0', tk.END)  # Limpar todo o conteúdo do console
    console_text.config(state=tk.DISABLED)  # Desabilitar novamente para impedir edição

    # Exibir mensagem de confirmação no console
    console_text.config(state=tk.NORMAL)  # Habilitar edição para adicionar a mensagem
    console_text.insert(tk.END, "Press run to start simulation.\n")
    console_text.config(state=tk.DISABLED)  # Desabilitar para impedir edição manual


def clear_fields_rev(graphics_frame_reviews):

    # Limpar labels de valores da aba Simulation
    rev_name_value.config(text="")
    rev_name_label.config(text="")
    rev_grids_value.config(text="")
    rev_episodes_value.config(text="")
    rev_steps_value.config(text="")
    
    rev_prey_inicial_value.config(text="")
    rev_predator_inicial_value.config(text="")
    rev_prey_limite_value.config(text="")
    rev_predator_limite_value.config(text="")
    
    rev_prey_reproduction_value.config(text="")
    rev_predator_reproduction_value.config(text="")
    rev_prey_fatigue_value.config(text="")
    rev_predator_fatigue_value.config(text="")
    
    rev_prey_learning_value.config(text="")
    rev_predator_learning_value.config(text="")
    rev_prey_design_value.config(text="")
    rev_predator_design_value.config(text="")
    rev_prey_attention_value.config(text="")
    rev_predator_attention_value.config(text="")

    initialize_empty_graphics(graphics_frame_reviews)


def save_data_to_json():
    try:

        # Primeiro, validar os campos de entrada
        if not validate_inputs():
            return  # Se a validação falhar, interrompa o processo de salvamento

        # Coletar os dados dos campos de entrada no formato solicitado
        data = {
            "model": model_name_entry.get(),
            "environment": {
                "grid_size": grids_entry.get(),
                "episodes": episodes_entry.get(),
                "steps": steps_entry.get()
            },
            "population": {
                "prey_initial_count": prey_inicial_entry.get(),
                "predator_initial_count": predator_inicial_entry.get(),
                "prey_max_count": prey_limite_entry.get(),
                "predator_max_count": predator_limite_entry.get(),
                "prey_spawn_rate": prey_reproduction_entry.get(),
                "predator_spawn_rate": predator_reproduction_entry.get(),
                "prey_step_decay": prey_fatigue_entry.get(),
                "predator_step_decay": predator_fatigue_entry.get()
            },
            "neural_network": {
                "prey_learning_model": prey_learning_var.get(),
                "predator_learning_model": predator_learning_var.get(),
                "prey_design_variant": prey_design_var.get(),
                "predator_design_variant": predator_design_var.get(),

            },
            "simulation_data": []
        }

        # Definindo o caminho correto dentro da pasta do projeto
        # Verifica se o diretório existe, senão cria
        save_path = os.path.join(os.path.dirname(__file__), 'save', 'models')
        os.makedirs(save_path, exist_ok=True)

        # Nome do arquivo será o nome do modelo
        filename = os.path.join(save_path, f"{data['model']}.json")

        # Salvando o dicionário em um arquivo JSON
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)


        # Exibir mensagem de sucesso
        messagebox.showinfo("Success", f"Data saved to {filename}")
    except Exception as e:
        # Exibir mensagem de erro e pedir para verificar o log
        print(f"Error saving data: {e}")
        messagebox.showerror("Error", "Failed to save data. Please check the log for more details.")


def exit_application():
    root.quit()


def load_graph_data_and_refresh_canvas(canvas_frame):
    try:
        # Definir o caminho inicial como 'save/simulations' com relação ao diretório do arquivo executado
        script_directory = os.path.dirname(os.path.abspath(__file__))  # Diretório do arquivo .py
        initial_directory = os.path.join(script_directory, 'save', 'models')  # Caminho completo para 'save/simulations'

        # Verificar se a pasta 'save/simulations' existe, caso contrário, mostrar erro
        if not os.path.exists(initial_directory):
            messagebox.showerror("Error", f"Directory not found: {initial_directory}")
            return

        # Abrir a caixa de diálogo no diretório 'save/simulations'
        file_path = filedialog.askopenfilename(
            initialdir=initial_directory,  # Definir a pasta inicial
            title="Open JSON File",
            filetypes=(("JSON Files", "*.json"), ("All Files", "*.*"))
        )

        if not file_path:
            return

        # Abrir o arquivo JSON e carregar os dados
        with open(file_path, "r") as json_file:
            data = json.load(json_file)

        # Primeiro, chama a função que carrega os dados de revisão
        read_review_data(data)  # Chama read_review_data e passa os dados JSON

        # Carregar dados de simulação e quantitativos
        df_data_sim = read_simulation_data(data)
        df_data_quantitative = read_quantitative_data(data)
        df_data_behavior = read_behavior_data(data)
        
        if df_data_sim is None or df_data_quantitative is None or df_data_behavior is None:
            messagebox.showerror("Error", "Simulation or quantitative data not loaded properly!")
            return

        # Atualiza o frame dos gráficos com os dados carregados
        setup_graphics_frame(canvas_frame, df_data_sim, df_data_quantitative, df_data_behavior)      

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load simulation data: {str(e)}")

def load_graph_data():
    global global_simulation_data  # Usar a variável global com os dados de simulação

    if global_simulation_data is None:
        messagebox.showerror("Error", "Simulation data not loaded! Please load the simulation data first.")
        return None

    # Processar os dados da simulação (pode ser ajustado conforme necessário)
    df = pd.DataFrame(global_simulation_data, columns=['episode', 'prey', 'predator'])
    
    return df

def render_graph(canvas, fig):
    for widget in canvas.winfo_children():  # Limpar o canvas antes de renderizar novo gráfico
        widget.destroy()
    fig.tight_layout()  # Ajusta o layout para evitar cortes
    canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    canvas_agg.draw()
    canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def setup_reviews_tab(notebook):
    reviews_tab = tk.Frame(notebook, bg="white")
    notebook.add(reviews_tab, text="Analyze")
    graphics_frame_reviews = tk.Frame(reviews_tab, bg="white")
    graphics_frame_reviews.place(relwidth=1, relheight=1)
    
    initialize_empty_graphics(graphics_frame_reviews)

def create_prey_population_graph(canvas, data, sma_percentage=0):
    # Converte `sma_percentage` para 0 se for `None`
    sma_percentage = sma_percentage or 0
    
    # Limpando o canvas antes de desenhar o novo gráfico
    for widget in canvas.winfo_children():
        widget.destroy()
    
    fig, ax = plt.subplots(figsize=(5, 3))

    print(f'entrou no grafico sma_percentage: {sma_percentage} ')

    # Verificar se todos os valores de 'prey' e 'episode' são zero
    if data['prey'].sum() == 0 or data['episode'].sum() == 0:
        ax.set_xlabel("Episodes", fontsize=9)
        ax.set_ylabel("Prey Population", fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
        ax.plot([], [])  # Plot vazio para garantir que o gráfico fique sem dados
    else:
        # Aplicar o SMA se uma porcentagem foi fornecida
        if sma_percentage > 0:
            # Convertendo o percentual em uma janela de SMA
            window_size = max(1, int(len(data) * sma_percentage))
            data_smoothed = data.rolling(window=window_size, center=True).mean()
            print(f"[LOG] Aplicando SMA com janela de {window_size} pontos")
        else:
            data_smoothed = data  # Usa os dados completos se o SMA não for aplicado
            print("[LOG] Exibindo dados completos sem SMA")

        # Plotar a série temporal de dados
        ax.plot(data_smoothed['episode'], data_smoothed['prey'], color='blue')
        ax.set_xlabel("Episodes", fontsize=9)
        ax.set_ylabel("Prey Population", fontsize=9)
        ax.tick_params(axis='both', labelsize=9)

    # Renderizar o gráfico no canvas
    render_graph(canvas, fig)

def create_predator_population_graph(canvas, data, sma_percentage=0):
    sma_percentage = sma_percentage or 0  # Define como 0 se None

    # Limpando o canvas antes de desenhar o novo gráfico
    for widget in canvas.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(5, 3))

    # Verificar se todos os valores de 'predator' e 'episode' são zero
    if data['predator'].sum() == 0 or data['episode'].sum() == 0:
        ax.set_xlabel("Episodes", fontsize=9)
        ax.set_ylabel("Predator Population", fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
        ax.plot([], [])  # Plot vazio para manter o gráfico sem dados
    else:
        # Aplicar o SMA se uma porcentagem foi fornecida
        if sma_percentage > 0:
            window_size = max(1, int(len(data) * sma_percentage))
            data_smoothed = data.rolling(window=window_size, center=True).mean()
            print(f"[LOG] Aplicando SMA com janela de {window_size} pontos")
        else:
            data_smoothed = data
            print("[LOG] Exibindo dados completos sem SMA")

        # Plotar a série temporal de dados
        ax.plot(data_smoothed['episode'], data_smoothed['predator'], color='red')
        ax.set_xlabel("Episodes", fontsize=9)
        ax.set_ylabel("Predator Population", fontsize=9)
        ax.tick_params(axis='both', labelsize=9)

    # Renderizar o gráfico no canvas
    render_graph(canvas, fig)

def create_prey_vs_predator_graph(canvas, data, sma_percentage=0):
    # Converte `sma_percentage` para 0 se for `None`
    sma_percentage = sma_percentage or 0
    
    # Limpando o canvas antes de desenhar o novo gráfico
    for widget in canvas.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(5, 3))
    print(f'entrou no grafico sma_percentage: {sma_percentage}')

    # Verificar se todos os valores de 'prey' e 'predator' são zero
    if data['prey'].sum() == 0 and data['predator'].sum() == 0:
        ax.set_xlabel("Prey Population", fontsize=9)
        ax.set_ylabel("Predator Population", fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
        ax.legend(["Data Points", "Linear Regression"], fontsize=8)
    else:
        # Usar `data` diretamente, pois `sma_percentage` foi aplicado no `on_combobox_select`
        data_to_plot = data  # usa os dados passados com ou sem SMA

        # Gráfico de dispersão entre prey e predator
        ax.scatter(data_to_plot['prey'], data_to_plot['predator'], color='green', label='Data Points')

        # Adicionar linha de tendência linear
        z = np.polyfit(data_to_plot['prey'].dropna(), data_to_plot['predator'].dropna(), 1)  # Ajuste linear
        p = np.poly1d(z)

        # Plotar a linha de tendência
        ax.plot(data_to_plot['prey'], p(data_to_plot['prey']), color='blue', linestyle='--', label='Linear Regression')

        # Configurar os eixos
        ax.set_xlabel("Prey Population", fontsize=9)
        ax.set_ylabel("Predator Population", fontsize=9)
        ax.tick_params(axis='both', labelsize=9)

        # Adicionar legenda
        ax.legend(fontsize=8)

    # Renderizar o gráfico no canvas
    render_graph(canvas, fig)

def create_prey_and_predator_population_graph(canvas, data, sma_percentage=0):
    sma_percentage = sma_percentage or 0  # Define como 0 se None

    # Limpando o canvas antes de desenhar o novo gráfico
    for widget in canvas.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(5, 3))

    # Verificar se todos os valores de 'prey', 'predator' e 'episode' são zero
    if data['prey'].sum() == 0 and data['predator'].sum() == 0 and data['episode'].sum() == 0:
        ax.set_xlabel("Episodes", fontsize=9)
        ax.set_ylabel("Population", fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
        ax.plot([], [], label='Prey', color='blue')
        ax.plot([], [], label='Predator', color='red')
        ax.legend(fontsize=9)
    else:
        # Aplicar o SMA se uma porcentagem foi fornecida
        if sma_percentage > 0:
            window_size = max(1, int(len(data) * sma_percentage))
            data_smoothed = data.rolling(window=window_size, center=True).mean()
            print(f"[LOG] Aplicando SMA com janela de {window_size} pontos")
        else:
            data_smoothed = data
            print("[LOG] Exibindo dados completos sem SMA")

        # Plotar as populações de presas e predadores
        ax.plot(data_smoothed['episode'], data_smoothed['prey'], label='Prey', color='blue')
        ax.plot(data_smoothed['episode'], data_smoothed['predator'], label='Predator', color='red')

        ax.set_xlabel("Episodes", fontsize=9)
        ax.set_ylabel("Population", fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
        ax.legend(fontsize=9)

    # Renderizar o gráfico no canvas
    render_graph(canvas, fig)

def create_quantitative_population(canvas, data_quant, data_behavior):
    # Limpa o canvas para evitar duplicação
    for widget in canvas.winfo_children():
        widget.destroy()

    # Verifica se `data_quant` é um DataFrame
    if not isinstance(data_quant, pd.DataFrame):
        messagebox.showerror("Error", "'data_quant' is not in the expected DataFrame format.")
        print("Unexpected data format:", type(data_quant))  # Imprime o tipo inesperado para diagnóstico
        return

    # Fonte consistente para dados quantitativos e comportamentais
    content_font = ("Arial", 8)  # Fonte padrão para o conteúdo regular
    bold_font = ("Arial", 8, "bold")  # Fonte em negrito para os comportamentos

    # Separar dados de `Predator` e `Prey` a partir da coluna 'Type'
    predator_data = data_quant[data_quant['Type'] == 'Predator'][['Reward', 'Done', 'Step']].values
    prey_data = data_quant[data_quant['Type'] == 'Prey'][['Reward', 'Done', 'Step']].values

    # Calcula as estatísticas para `predator` e `prey`
    predator_stats = calculate_statistics(predator_data)
    prey_stats = calculate_statistics(prey_data)

    # Configuração dos frames para Prey e Predator
    prey_frame = tk.LabelFrame(canvas, text="Prey", bg="white")
    prey_frame.grid(row=0, column=0, padx=5, pady=1, sticky="nsew")

    predator_frame = tk.LabelFrame(canvas, text="Predator", bg="white")
    predator_frame.grid(row=0, column=1, padx=5, pady=1, sticky="nsew")

    # Adiciona dados quantitativos ao frame Prey
    add_data_to_frame(prey_frame, prey_stats, "Reward", "Done", "Step")
    # Adiciona dados quantitativos ao frame Predator
    add_data_to_frame(predator_frame, predator_stats, "Reward", "Done", "Step")

    # Criação do LabelFrame para comportamentos de Prey e Predator
    prey_behavior_frame = tk.LabelFrame(canvas, text="Prey Behavior", bg="white")
    prey_behavior_frame.grid(row=1, column=0, padx=5, pady=1, sticky="nsew")

    predator_behavior_frame = tk.LabelFrame(canvas, text="Predator Behavior", bg="white")
    predator_behavior_frame.grid(row=1, column=1, padx=5, pady=1, sticky="nsew")

    # Filtra os dados de comportamento para `Prey` e `Predator`
    prey_behavior = data_behavior[data_behavior['Type'] == 'Prey']
    predator_behavior = data_behavior[data_behavior['Type'] == 'Predator']

    # Adiciona dados de comportamento ao frame Prey Behavior em formato de tabela com alinhamento ajustado
    for i, (_, row) in enumerate(prey_behavior.iterrows()):
        # Frame intermediário para alinhar o conteúdo em duas colunas
        row_frame = tk.Frame(prey_behavior_frame, bg="white")
        row_frame.grid(row=i, column=0, sticky="ew")

        # Configurações de alinhamento para as Labels
        behavior_label = tk.Label(row_frame, text=row['Behavior'], bg="white", font=bold_font)  # Negrito
        count_label = tk.Label(row_frame, text=row['Count'], bg="white", font=content_font)  # Regular

        behavior_label.grid(row=0, column=0, sticky="w", padx=10)
        count_label.grid(row=0, column=1, sticky="e", padx=(40, 10))

        # Configuração para expandir a segunda coluna do frame intermediário para a direita
        row_frame.grid_columnconfigure(0, weight=1)
        row_frame.grid_columnconfigure(1, weight=1)

    # Adiciona dados de comportamento ao frame Predator Behavior em formato de tabela com alinhamento ajustado
    for i, (_, row) in enumerate(predator_behavior.iterrows()):
        # Frame intermediário para alinhar o conteúdo em duas colunas
        row_frame = tk.Frame(predator_behavior_frame, bg="white")
        row_frame.grid(row=i, column=0, sticky="ew")

        # Configurações de alinhamento para as Labels
        behavior_label = tk.Label(row_frame, text=row['Behavior'], bg="white", font=bold_font)  # Negrito
        count_label = tk.Label(row_frame, text=row['Count'], bg="white", font=content_font)  # Regular

        behavior_label.grid(row=0, column=0, sticky="w", padx=10)
        count_label.grid(row=0, column=1, sticky="e", padx=(40, 10))

        # Configuração para expandir a segunda coluna do frame intermediário para a direita
        row_frame.grid_columnconfigure(0, weight=1)
        row_frame.grid_columnconfigure(1, weight=1)

    # Configuração das colunas para expandir corretamente
    canvas.grid_columnconfigure(0, weight=1)
    canvas.grid_columnconfigure(1, weight=1)

def create_histogram_population(canvas, data, combobox_sma):
    # Força o combobox SMA para 0% para este gráfico
    combobox_sma.set("0%")

    fig, ax = plt.subplots(figsize=(5, 3))  # Tamanho padrão
    prey_population = data['prey']
    predator_population = data['predator']

    # Verificar se os valores de 'prey' e 'predator' são todos zero
    if prey_population.sum() == 0 and predator_population.sum() == 0:
        ax.set_xlabel("Population (%)", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
        ax.legend(["Prey Population", "Predator Population"], fontsize=7)
    else:
        ax.hist(prey_population, bins=10, color='blue', alpha=0.7, label="Prey Population")
        ax.hist(predator_population, bins=10, color='red', alpha=0.7, label="Predator Population")
        ax.set_xlabel("Population (%)", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
        ax.legend(fontsize=7)

    # Renderizar o gráfico no canvas
    render_graph(canvas, fig)
    plt.close(fig)

def add_data_to_frame(frame, data, column1, column2, column3):
    # Adiciona cabeçalho das colunas com centralização
    tk.Label(frame, text="", font=("Arial", 8, "bold"), bg="white").grid(row=0, column=0, padx=10, pady=1, sticky="nsew")
    tk.Label(frame, text=column1, font=("Arial", 8, "bold"), bg="white").grid(row=0, column=1, padx=10, pady=1, sticky="nsew")
    tk.Label(frame, text=column2, font=("Arial", 8, "bold"), bg="white").grid(row=0, column=2, padx=10, pady=1, sticky="nsew")
    tk.Label(frame, text=column3, font=("Arial", 8, "bold"), bg="white").grid(row=0, column=3, padx=10, pady=1, sticky="nsew")
    
    # Adiciona os valores para cada linha estatística com centralização
    stats_labels = ["Mean", "Median", "SD", "Max", "Min", "Variance", "Range", "IQR"]
    for row, label in enumerate(stats_labels, start=1):
        tk.Label(frame, text=label, font=("Arial", 8, "bold"), bg="white").grid(row=row, column=0, padx=10, pady=1, sticky="w")
        tk.Label(frame, text=f"{data[column1][label]:.2f}", font=("Arial", 8), bg="white").grid(row=row, column=1, padx=10, pady=1, sticky="nsew")
        tk.Label(frame, text=f"{data[column2][label]:.2f}", font=("Arial", 8), bg="white").grid(row=row, column=2, padx=10, pady=1, sticky="nsew")
        tk.Label(frame, text=f"{data[column3][label]:.2f}", font=("Arial", 8), bg="white").grid(row=row, column=3, padx=10, pady=1, sticky="nsew")

    # Configura as colunas para expandirem igualmente, ajudando na centralização
    for col in range(4):
        frame.grid_columnconfigure(col, weight=1)

def add_status_to_frame(frame, status_list):
    # Adiciona os status com fundo branco, utilizando widgets de `tk`
    for idx, status in enumerate(status_list):
        tk.Label(frame, text=status, font=("Arial", 8), bg="white").grid(row=idx, column=0, padx=5, pady=1, sticky="w")

def calculate_statistics(data):
    """Calcula estatísticas para uma lista de tuplas (reward, done, step) e limita os resultados a duas casas decimais"""
    

    rewards = np.array([row[0] for row in data])
    dones = np.array([row[1] for row in data])
    steps = np.array([row[2] for row in data])

    stats = {
        "Reward": {
            "Mean": round(np.mean(rewards), 2),
            "Median": round(np.median(rewards), 2),
            "SD": round(np.std(rewards), 2),
            "Max": round(np.max(rewards), 2),
            "Min": round(np.min(rewards), 2),
            "Variance": round(np.var(rewards), 2),
            "Range": round(np.ptp(rewards), 2),
            "IQR": round(iqr(rewards), 2)
        },
        "Done": {
            "Mean": round(np.mean(dones), 2),
            "Median": round(np.median(dones), 2),
            "SD": round(np.std(dones), 2),
            "Max": round(np.max(dones), 2),
            "Min": round(np.min(dones), 2),
            "Variance": round(np.var(dones), 2),
            "Range": round(np.ptp(dones), 2),
            "IQR": round(iqr(dones), 2)
        },
        "Step": {
            "Mean": round(np.mean(steps), 2),
            "Median": round(np.median(steps), 2),
            "SD": round(np.std(steps), 2),
            "Max": round(np.max(steps), 2),
            "Min": round(np.min(steps), 2),
            "Variance": round(np.var(steps), 2),
            "Range": round(np.ptp(steps), 2),
            "IQR": round(iqr(steps), 2)
        }
    }
    return stats

def create_stacked_area_graph(canvas, data):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.stackplot(data.index + 1, data['prey'], data['predator'], labels=['Prey', 'Predator'], colors=['blue', 'red'])
    ax.set_xlabel("Episodes", fontsize=9)
    ax.set_ylabel("Population", fontsize=9)
    ax.tick_params(axis='both', labelsize=9)
    ax.legend(loc='upper right', fontsize=7)
    render_graph(canvas, fig)

def create_population_boxplot(canvas, data):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.boxplot([data['prey'], data['predator']], labels=['Prey', 'Predator'], patch_artist=True)
    ax.set_xlabel("Population Type", fontsize=9)
    ax.set_ylabel("Population Size", fontsize=9)
    ax.tick_params(axis='both', labelsize=9)
    render_graph(canvas, fig)

# Inicialização da variável global para SMA
global_combobox_sma = None

def on_combobox_select(event, canvas, data_sim, data_quant, data_behavior, selected_value="Quantitative Population Analysis", sma_value="0%"):
    """Função de callback para seleção de gráfico com opção de SMA."""
    print(f"[LOG] Gráfico Selecionado: {selected_value}")
    print(f"[LOG] Valor de SMA Selecionado: {sma_value}")

    # Converte o valor de SMA para uma porcentagem
    sma_percentage = float(sma_value.strip('%')) / 100 if sma_value.strip('%').isdigit() else 0.0
    print(f"[LOG] Valor de SMA Convertido: {sma_percentage}")

    # Desativa o SMA para gráficos que não utilizam essa configuração
    if selected_value in ["Quantitative Population Analysis", "Histogram of Populations", "Population Box Plot"]:
        sma_percentage = 0.0
        if global_combobox_sma:
            global_combobox_sma.set("0%")
        print(f"[LOG] Desativando SMA para {selected_value}")

    # Renderiza o gráfico apropriado com base na seleção do usuário
    if selected_value == "Quantitative Population Analysis":
        create_quantitative_population(canvas, data_quant, data_behavior)
    elif selected_value == "Histogram of Populations":
        create_histogram_population(canvas, data_sim, global_combobox_sma)
    elif selected_value == "Population Box Plot":
        create_population_boxplot(canvas, data_sim)

    elif selected_value == "Prey vs Predator Population":
        data_to_plot = data_sim.rolling(window=int(len(data_sim) * sma_percentage), center=True).mean() if sma_percentage else data_sim
        create_prey_vs_predator_graph(canvas, data_to_plot, sma_percentage=sma_percentage)
    
    elif selected_value == "Prey Population per Episode":
        data_to_plot = data_sim.rolling(window=int(len(data_sim) * sma_percentage), center=True).mean() if sma_percentage else data_sim
        create_prey_population_graph(canvas, data_to_plot, sma_percentage=sma_percentage)
    
    elif selected_value == "Predator Population per Episode":
        data_to_plot = data_sim.rolling(window=int(len(data_sim) * sma_percentage), center=True).mean() if sma_percentage else data_sim
        create_predator_population_graph(canvas, data_to_plot)
    
    elif selected_value == "Prey and Predator Population per Episode":
        data_to_plot = data_sim.rolling(window=int(len(data_sim) * sma_percentage), center=True).mean() if sma_percentage else data_sim
        create_prey_and_predator_population_graph(canvas, data_to_plot)

def setup_graphics_frame(graphics_frame_reviews, data_sim, data_quant, data_behavior):
    """Configura a interface dos comboboxes e o layout da área de gráficos."""
    global global_combobox_sma  # Para uso consistente na interface

    # 1. Frame para Combobox
    combobox_frame = tk.Frame(graphics_frame_reviews, bg="white")
    combobox_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    combobox_frame.grid_columnconfigure(1, minsize=250)  # Define um tamanho mínimo para o Combobox de Assessment

    # 2. Label e Combobox para seleção de gráficos
    label = tk.Label(combobox_frame, text="Simulation Assessment:", anchor="w", bg="white")
    label.grid(row=0, column=0, sticky="w", padx=(0, 0))
    combobox = ttk.Combobox(combobox_frame, state="readonly", width=40)
    combobox.grid(row=0, column=1, sticky="w", padx=(0, 0))
    graph_options = [
        "Quantitative Population Analysis",
        "Prey Population per Episode",
        "Predator Population per Episode",
        "Prey and Predator Population per Episode",
        "Prey vs Predator Population",
        "Histogram of Populations",
        "Population Box Plot",
    ]
    combobox['values'] = graph_options
    combobox.current(0)  # Define a primeira opção como selecionada por padrão

    # 3. Label e Combobox para SMA
    label_sma = tk.Label(combobox_frame, text="SMA %:", anchor="w", bg="white")
    label_sma.grid(row=0, column=2, sticky="e", padx=5)
    global_combobox_sma = ttk.Combobox(combobox_frame, state="readonly", values=["0%", "10%", "15%", "20%", "25%", "30%", "40%", "50%"], width=5)
    global_combobox_sma.set("0%")  # Define o valor padrão
    global_combobox_sma.grid(row=0, column=3, sticky="e", padx=5)

    # 4. Frame de exibição dos gráficos
    canvas_frame = tk.Frame(graphics_frame_reviews, bg="white")
    canvas_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    graphics_frame_reviews.grid_rowconfigure(1, weight=1)  # Permite que o frame de gráficos expanda
    graphics_frame_reviews.grid_columnconfigure(0, weight=1)

    # 5. Eventos de seleção para os Comboboxes
    combobox.bind("<<ComboboxSelected>>", lambda event: on_combobox_select(
        event, canvas_frame, data_sim, data_quant, data_behavior, selected_value=combobox.get(), sma_value=global_combobox_sma.get()
    ))
    global_combobox_sma.bind("<<ComboboxSelected>>", lambda event: on_combobox_select(
        None, canvas_frame, data_sim, data_quant, data_behavior, selected_value=combobox.get(), sma_value=global_combobox_sma.get()
    ))

    # Exibe o gráfico padrão ao inicializar
    on_combobox_select(None, canvas_frame, data_sim, data_quant, data_behavior, selected_value=combobox.get(), sma_value=global_combobox_sma.get())

def initialize_empty_graphics(graphics_frame_reviews):
    """Inicializa os dados padrão e configura o frame de gráficos."""
    # Dados iniciais para exibição
    data_sim = pd.DataFrame([[0, 0, 0]], columns=['episode', 'prey', 'predator'])
    data_quant = pd.DataFrame([["Predator", 0, 0, 0], ["Prey", 0, 0, 0]], columns=["Type", "Reward", "Done", "Step"])
    data_behavior = pd.DataFrame([
        ["Predator", "Prey captured", 0],
        ["Predator", "Nearby prey", 0],
        ["Predator", "Exploring map", 0],
        ["Prey", "Predator escape", 0],
        ["Prey", "Nearby predator", 0],
        ["Prey", "Exploring map", 0]
    ], columns=["Type", "Behavior", "Count"])

    # Configura o layout e Comboboxes para gráficos
    setup_graphics_frame(graphics_frame_reviews, data_sim, data_quant, data_behavior)

def center_window(root, width=900, height=600):
    # Atualizar a geometria da janela antes de calcular sua posição
    root.update_idletasks()  

    # Obter a largura e altura da tela
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calcular a posição x e y para centralizar a janela
    position_x = (screen_width // 2) - (width // 2)
    position_y = (screen_height // 2) - (height // 2)

    # Definir a geometria da janela
    root.geometry(f'{width}x{height}+{position_x}+{position_y}')
    # Obter a largura e altura da tela
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calcular a posição x e y para centralizar a janela
    position_x = (screen_width // 2) - (width // 2)
    position_y = (screen_height // 2) - (height // 2)

    # Definir a geometria da janela
    root.geometry(f'{width}x{height}+{position_x}+{position_y}')

class ToolTip(object):
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        if self.tip_window or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#f0f0f0", relief=tk.SOLID, borderwidth=0,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tooltip(self, event):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

def verify_report_in_review():
    # Verifica se rev_name_value está vazia
    if not rev_name_value.cget("text"):
        print("[LOG] rev_name_value está vazia.")
        messagebox.showerror("Error", "No model available for report.")
        return None

    # Obtém o nome do modelo a partir de rev_name_value
    model_name = rev_name_value.cget("text")

    # Captura o caminho completo do diretório 'save/models'
    script_directory = os.path.dirname(os.path.abspath(__file__))  # Diretório do arquivo .py
    initial_directory = os.path.join(script_directory, 'save', 'models')  # Caminho completo para 'save/models'
    json_file_path = os.path.join(initial_directory, f"{model_name}.json")


    # Verifica se o arquivo existe
    if not os.path.exists(json_file_path):
        messagebox.showerror("Error", "The model file does not exist.")
        return None

    try:
        # Carrega o conteúdo do arquivo JSON
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        # Verifica se as chaves 'simulation_data', 'quantitative_data' e 'behavior_data' existem e não estão vazias
        if not data.get('simulation_data') or not data['simulation_data']:
            messagebox.showerror("Error", "The 'simulation_data' field is missing or empty.")
            return None
        if not data.get('quantitative_data') or not data['quantitative_data']:
            messagebox.showerror("Error", "The 'quantitative_data' field is missing or empty.")
            return None
        if not data.get('behavior_data') or not data['behavior_data']:
            messagebox.showerror("Error", "The 'behavior_data' field is missing or empty.")
            return None

        # Retorna o nome do arquivo
        return f"{model_name}.json"

    except Exception as e:
        messagebox.showerror("Error", f"Failed to read the model file: {str(e)}")
        return None
    
def verify_monitor_in_simulation():
    try:
        # Verifica se sim_name_value está vazia
        print("[LOG] Verificando se sim_name_value está vazia...")
        if not sim_name_value.cget("text"):
            print("[LOG] sim_name_value está vazia.")
            messagebox.showerror("Error", "No model available for simulation monitor.")
            return None

        # Obtém o nome do modelo a partir de sim_name_value
        model_name = sim_name_value.cget("text")


        name_sim = "sim_" + model_name + ".csv"

        print(f"[LOG] Nome do arquivo csv: {name_sim}")
        return name_sim

    except Exception as e:
        print(f"[LOG] Erro ao ler o arquivo: {str(e)}")
        messagebox.showerror("Error", f"Failed to read the model file: {str(e)}")
        return None

def delete_file(directory, filename):

    file_path = os.path.join(directory, filename)

    if os.path.exists(file_path):
        os.remove(file_path)

# Função que chama diretamente a visualização sem uso de threads
def iniciar_visualizacao(nome_do_arquivo_csv):
    if not nome_do_arquivo_csv:
        print("Erro: Nenhum nome de arquivo CSV fornecido.")
        return

    print(f"[LOG] Nome do modelo obtido: {nome_do_arquivo_csv}")

    # Chama diretamente a função de visualização
    preparar_e_chamar_visualizacao(nome_do_arquivo_csv)

def start_simulation():
    global slider_label  # Definindo slider_label como global para uso na função on_slider_change    
    global model_name_entry, grids_entry, episodes_entry, steps_entry  # Tornar os campos globais para salvar
    global prey_inicial_entry, predator_inicial_entry, prey_limite_entry, predator_limite_entry  # Campos da aba direita
    global prey_reproduction_entry, predator_reproduction_entry, prey_fatigue_entry, predator_fatigue_entry
    global prey_attention_var, predator_attention_var
    global listbox_neural_network  # Definindo listbox como global para ser acessado em outras funções
    global save_button_image  # Manter referência às imagens para o garbage collector
    global console  # Adicionar o console como global se você for usar em outra parte do código
    global model_name_value, grids_value, episodes_value, steps_value, neural_network_value
    global prey_inicial_value, predator_inicial_value
    global prey_limite_value, predator_limite_value
    global prey_reproduction_value, predator_reproduction_value
    global prey_fatigue_value, predator_fatigue_value
    global prey_attention_value, predator_attention_value
    global model_name_entry, grids_entry, episodes_entry, steps_entry
    global prey_inicial_entry, predator_inicial_entry, prey_limite_entry, predator_limite_entry
    global prey_reproduction_entry, predator_reproduction_entry, prey_fatigue_entry, predator_fatigue_entry
    global prey_learning_var, predator_learning_var, prey_design_var, predator_design_var
    global prey_learning_menu, predator_learning_menu
    global prey_design_menu, predator_design_menu
    global prey_learning_value, predator_learning_value, prey_design_value, predator_design_value, run_button_image, open_button_image_sim, run_button_image_sim, cancel_button_image_sim, run_button_sim, cancel_button_simulation, run_button_simulation
    global sim_name_label, sim_name_value, sim_separator_top, simulation_label_frame, sim_grids_label, sim_grids_value, sim_episodes_label, sim_episodes_value, sim_episodes_label, sim_episodes_value, sim_steps_label, sim_steps_value, sim_separator_bottom, sim_population_frame, sim_prey_label, sim_predator_label, sim_inicial_label, sim_prey_inicial_value, sim_predator_inicial_value, sim_limite_label, sim_prey_limite_value, sim_predator_limite_value, sim_reproduction_label, sim_prey_reproduction_value, sim_predator_reproduction_value, sim_fatigue_label, sim_prey_fatigue_value, sim_predator_fatigue_value, sim_learning_label, sim_prey_learning_value, sim_predator_learning_value, sim_design_label, sim_prey_design_value, sim_predator_design_value, sim_attention_label, sim_prey_attention_value, sim_predator_attention_value
    global rev_name_label, rev_name_value, rev_separator_top, simulation_label_frame, rev_grids_label, rev_grids_value, rev_episodes_label, rev_episodes_value, rev_episodes_label, rev_episodes_value, rev_steps_label, rev_steps_value, rev_separator_bottom, rev_population_frame, rev_prey_label, rev_predator_label, rev_inicial_label, rev_prey_inicial_value, rev_predator_inicial_value, rev_limite_label, rev_prey_limite_value, rev_predator_limite_value, rev_reproduction_label, rev_prey_reproduction_value, rev_predator_reproduction_value, rev_fatigue_label, rev_prey_fatigue_value, rev_predator_fatigue_value, rev_learning_label, rev_prey_learning_value, rev_predator_learning_value, rev_design_label, rev_prey_design_value, rev_predator_design_value, rev_attention_label, rev_prey_attention_value, rev_predator_attention_value
    global console_text
    global combobox_sma

    for widget in root.winfo_children():
        if widget != header_frame:
            widget.destroy()

    # Criar um Notebook com 3 abas e preencher toda a janela
    notebook = ttk.Notebook(root)
    notebook.place(x=0, y=115, width=900, height=485)
    

    # Criando as abas
    models_tab = tk.Frame(notebook, bg="white")  # Fundo branco para o conteúdo das abas
    simulation_tab = tk.Frame(notebook, bg="white")
    reviews_tab = tk.Frame(notebook, bg="white")

    # Definir tamanhos fixos para os frames (conteúdo das abas)
    models_tab.place(width=900, height=455)
    simulation_tab.place(width=900, height=455)
    reviews_tab.place(width=900, height=455)

    # Adicionando as abas ao notebook
    notebook.add(models_tab, text="Models")
    notebook.add(simulation_tab, text="Simulate")
    notebook.add(reviews_tab, text="Analyze")
    


    # Estilizar as abas usando ttkthemes
    style = ttk.Style()
    style.theme_use('xpnative')  # Usar o tema 'default', 'alt', 'default', 'classic', 'xpnative'

    # Configurar as cores (tentativa)
    style.configure('TNotebook', background='#FFFFFF')  # Cor do fundo do notebook
    style.configure('TNotebook.Tab', font=('Arial Black', 10, 'bold'), padding=[20, 10], background='#FFFFFF')
    style.map('TNotebook.Tab', background=[('selected', '#FFFFFF')], foreground=[('selected', 'blue')])

    # Criar estilo para o Slider com cor branca
    style.configure("TScale", background="white")  # Define o fundo do slider como branco


    # =============================================================================================================================================
    # ABA MODELS
    # =============================================================================================================================================

    # Definir altura e largura total da janela
    window_width = 900
    window_height = 600

    # Cálculo para definir a largura dos frames proporcionalmente
    left_frame_width = int(window_width * 0.40)  # 40% da largura da janela
    right_frame_width = int(window_width * 0.60)  # 60% da largura da janela

    # Ajustar a altura total, subtraindo o espaço para botões e margens
    frame_height = window_height - 115  # Altura disponível após subtrair o espaço da barra de abas e cabeçalho

    # Configuração das colunas principais no reviews_tab (mesma proporção que a aba Simulation)
    models_tab.grid_columnconfigure(0, weight=1)  # Coluna esquerda
    models_tab.grid_columnconfigure(1, weight=0)  # Separador
    models_tab.grid_columnconfigure(2, weight=1)  # Coluna direita

    # Criar o left_frame (40% da largura) com altura ajustada para a aba Reviews
    models_left_frame = tk.Frame(models_tab, bg="white", width=left_frame_width, height=frame_height)
    models_left_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    models_left_frame.grid_columnconfigure(0, weight=1)  # Faz a coluna 0 expandir e se centralizar
    models_left_frame.grid_propagate(False)  # Impedir que o frame se redimensione automaticamente

    # Criar o right_frame (60% da largura) com altura ajustada para a aba Reviews
    models_right_frame = tk.Frame(models_tab, bg="white", width=right_frame_width, height=frame_height)
    models_right_frame.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)
    models_right_frame.grid_propagate(False)  # Impede que o frame se ajuste automaticamente ao conteúdo
    
    # Criar um LabelFrame para Nome Model
    model_frame = tk.LabelFrame(models_left_frame, text="Model", font=("Arial", 11, "bold"), bg="white", padx=10, pady=5)
    model_frame.grid(row=0, column=0, pady=5, padx=10, sticky="nsew", columnspan=2)

    # Configurar as colunas dentro do model_frame
    model_frame.grid_columnconfigure(0, weight=0)  # Coluna para os labels (Name)
    model_frame.grid_columnconfigure(1, weight=1)  # Coluna para o campo de entrada (Model)

    # Label para Model (alinhado à esquerda)
    model_name_label = tk.Label(model_frame, text="Name:", font=("Arial", 10, "bold"), bg="white", cursor="hand2")
    model_name_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")  # Alinhado à esquerda
    ToolTip(model_name_label, "Enter the name of the model for identification.")

    # Campo de entrada para Model
    model_name_entry = tk.Entry(model_frame, font=("Verdana", 10), justify="left", width=25)
    model_name_entry.grid(row=0, column=1, padx=10, pady=5, sticky="e")

    # Criar um LabelFrame para Grids, Episodes e Steps (Tabela do Environment)
    environment_frame = tk.LabelFrame(models_left_frame, text="Environment", font=("Arial", 11, "bold"), bg="white", padx=10, pady=5)
    environment_frame.grid(row=2, column=0, pady=5, padx=10, sticky="nsew", columnspan=2)

    # Configurar as colunas dentro do environment_frame
    environment_frame.grid_columnconfigure(0, weight=0)  # Coluna para os labels (Grid Size, Episodes, Steps)
    environment_frame.grid_columnconfigure(1, weight=1)  # Coluna para os campos de entrada

    # Label para Grid Size (alinhado à esquerda)
    grids_label = tk.Label(environment_frame, text="Grid Size:", font=("Arial", 10, "bold"), fg="black", bg="white", cursor="hand2")
    grids_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")  # Alinhado à esquerda
    ToolTip(grids_label, "Defines the length of one side of the grid, where the grid is a square.")

    # Campo de entrada para Grid Size (diminuído e alinhado à esquerda)
    grids_entry = tk.Entry(environment_frame, font=("Verdana", 10), justify="center", width=7)
    grids_entry.grid(row=0, column=1, padx=7, pady=5, sticky="e")

    # Label para Episodes (alinhado à esquerda)
    episodes_label = tk.Label(environment_frame, text="Episodes:", font=("Arial", 10, "bold"), bg="white", cursor="hand2")
    episodes_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")  # Alinhado à esquerda
    ToolTip(episodes_label, "Specifies the number of episodes the simulation will run.")

    # Campo de entrada para Episodes (diminuído e alinhado à esquerda)
    episodes_entry = tk.Entry(environment_frame, font=("Verdana", 10), justify="center", width=7)
    episodes_entry.grid(row=1, column=1, padx=7, pady=5, sticky="e")

    # Label para Steps (alinhado à esquerda)
    steps_label = tk.Label(environment_frame, text="Steps:", font=("Arial", 10, "bold"), bg="white", cursor="hand2")
    steps_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")  # Alinhado à esquerda
    ToolTip(steps_label, "Indicates the number of steps in each episode of the simulation.")

    # Campo de entrada para Steps (diminuído e alinhado à esquerda)
    steps_entry = tk.Entry(environment_frame, font=("Verdana", 10), justify="center", width=7)
    steps_entry.grid(row=2, column=1, padx=7, pady=5, sticky="e")

    # Criar um LabelFrame para Prey e Predator (Tabela de População)
    population_frame = tk.LabelFrame(models_left_frame, text="Population", font=("Arial", 11, "bold"), bg="white", padx=10, pady=5)
    population_frame.grid(row=9, column=0, columnspan=2, padx=10, pady=(5, 5), sticky="nsew")

    # Configurar as colunas para distribuição uniforme
    population_frame.grid_columnconfigure(0, weight=0)  # Coluna para labels (Initial Count, Max Count, etc.)
    population_frame.grid_columnconfigure(1, weight=1)  # Coluna para Prey
    population_frame.grid_columnconfigure(2, weight=1)  # Coluna para Predator

    # Cabeçalhos Prey e Predator
    prey_label = tk.Label(population_frame, text="Prey", font=("Arial", 10, "bold"), bg="white", width=10)
    prey_label.grid(row=0, column=1, padx=(2, 2), pady=(2, 5), sticky="e")  # Reduzir o padx para diminuir espaço lateral

    predator_label = tk.Label(population_frame, text="Predator", font=("Arial", 10, "bold"), bg="white", width=10)
    predator_label.grid(row=0, column=2, padx=(2, 2), pady=(2, 5), sticky="e")  # Reduzir o padx para diminuir espaço lateral

    # Labels e entradas para Initial Count
    inicial_label = tk.Label(population_frame, text="Initial Count", font=("Arial", 10, "bold"), bg="white", justify="left", cursor="hand2")
    inicial_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")  # Garantir alinhamento à esquerda
    ToolTip(inicial_label, "Specifies the initial number of agents at the start of the simulation.")

    # Ajustar a largura fixa e centralizar
    prey_inicial_entry = tk.Entry(population_frame, font=("Verdana", 10), bg="white", width=5, justify="center")
    prey_inicial_entry.grid(row=1, column=1, padx=5, pady=5)

    predator_inicial_entry = tk.Entry(population_frame, font=("Verdana", 10), bg="white", width=5, justify="center")
    predator_inicial_entry.grid(row=1, column=2, padx=5, pady=5)

    # Labels e entradas para Max Count
    limite_label = tk.Label(population_frame, text="Max Count", font=("Arial", 10, "bold"), bg="white", justify="left", cursor="hand2")
    limite_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")  # Garantir alinhamento à esquerda
    ToolTip(limite_label, "Defines the maximum number of agents allowed in the simulation.")

    prey_limite_entry = tk.Entry(population_frame, font=("Verdana", 10), bg="white", width=5, justify="center")
    prey_limite_entry.grid(row=2, column=1, padx=5, pady=5)

    predator_limite_entry = tk.Entry(population_frame, font=("Verdana", 10), bg="white", width=5, justify="center")
    predator_limite_entry.grid(row=2, column=2, padx=5, pady=5)

    # Labels e entradas para Spawn Rate
    reproduction_label = tk.Label(population_frame, text="Spawn Rate", font=("Arial", 10, "bold"), bg="white", justify="left", cursor="hand2")
    reproduction_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")  # Garantir alinhamento à esquerda
    ToolTip(reproduction_label, "Percentage chance of agent reproduction.")

    prey_reproduction_entry = tk.Entry(population_frame, font=("Verdana", 10), bg="white", width=5, justify="center")
    prey_reproduction_entry.grid(row=3, column=1, padx=5, pady=5)

    # Colocar o símbolo % como valor inicial
    prey_reproduction_entry.insert(0, "%")

    predator_reproduction_entry = tk.Entry(population_frame, font=("Verdana", 10), bg="white", width=5, justify="center")
    predator_reproduction_entry.grid(row=3, column=2, padx=5, pady=5)
    # Colocar o símbolo % como valor inicial
    predator_reproduction_entry.insert(0, "%")

    # Labels e entradas para Step Decay
    fatigue_label = tk.Label(population_frame, text="Step Decay", font=("Arial", 10, "bold"), bg="white", justify="left", cursor="hand2")
    fatigue_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")  # Garantir alinhamento à esquerda
    ToolTip(fatigue_label, "Amount of life points lost by agents for each step taken.")

    prey_fatigue_entry = tk.Entry(population_frame, font=("Verdana", 10), bg="white", width=5, justify="center")
    prey_fatigue_entry.grid(row=4, column=1, padx=5, pady=5)

    predator_fatigue_entry = tk.Entry(population_frame, font=("Verdana", 10), bg="white", width=5, justify="center")
    predator_fatigue_entry.grid(row=4, column=2, padx=5, pady=5)


    # Garantir que o frame principal `reviews_right_frame` expanda corretamente
    models_right_frame.grid_columnconfigure(0, weight=1)
    models_right_frame.grid_rowconfigure(0, weight=1)  
################################################################################################################################################
    # Criar o LabelFrame para Machine Learning
    neural_network_frame_models = tk.LabelFrame(models_right_frame, text="Neural Network", font=("Arial", 11, "bold"), bg="white", padx=10, pady=0)
    neural_network_frame_models.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)

    # Configurar as colunas e linhas dentro do LabelFrame
    neural_network_frame_models.grid_columnconfigure(0, weight=1)
    neural_network_frame_models.grid_columnconfigure(1, weight=0)
    neural_network_frame_models.grid_rowconfigure(0, weight=1)

    # Texto introdutório sobre os modelos de aprendizado
    intro_description = tk.Label(neural_network_frame_models,
        text=(
            "The models below are based on Deep Q-Network (DQN), a reinforcement learning\n"
            "algorithm that uses neural networks to approximate the Q-function.\n"
            "These models introduce enhancements to address specific limitations of standard DQN."
        ), font=("Verdana", 8), fg="black", bg="white", justify="left", wraplength=480)
    intro_description.grid(row=0, column=0, columnspan=4, padx=5, pady=(5, 0), sticky="w")

    # Subtítulo para Learning Model
    learning_subtitle = tk.Label(neural_network_frame_models, text="Learning Model", font=("Arial", 10, "bold"), fg="black", bg="white")
    learning_subtitle.grid(row=1, column=0, columnspan=4, padx=5, pady=(0, 0), sticky="w")

    # Descrição técnica dos Learning Models
    learning_description = tk.Label(neural_network_frame_models,
        text=(
            "DQN: Standard model using a neural network to approximate the Q-function. \n"
            "PER: Introduces Prioritized Experience Replay, prioritizing TD errors.  \n"
            "DUELING: Separates the Q-function into state value and action advantage. \n"
            "DOUBLE: Uses two networks to decouple actions and estimates, reducing bias.\n"
        ), font=("Verdana", 8), fg="black", bg="white", justify="left", wraplength=480)
    learning_description.grid(row=2, column=0, columnspan=4, padx=5, pady=(0, 0), sticky="w")

    # Labels e Comboboxes para Prey e Predator Learning Models
    prey_learning_label = tk.Label(neural_network_frame_models, text="Prey:", font=("Verdana", 10), bg="white")
    prey_learning_label.grid(row=3, column=0, padx=5, pady=(0, 0), sticky="e")
    prey_learning_var = tk.StringVar()
    prey_learning_var.set("")  # Valor inicial

    prey_learning_menu = ttk.Combobox(neural_network_frame_models, textvariable=prey_learning_var, values=["DQN", "PER", "DUELING", "DOUBLE"], width=10)
    prey_learning_menu.grid(row=3, column=1, padx=10, pady=(0, 0), sticky="w")



    predator_learning_label = tk.Label(neural_network_frame_models, text="Predator:", font=("Verdana", 10), bg="white")
    predator_learning_label.grid(row=3, column=2, padx=5, pady=(0, 0), sticky="e")

    predator_learning_var = tk.StringVar()
    predator_learning_var.set("")  # Valor inicial
    predator_learning_menu = ttk.Combobox(neural_network_frame_models, textvariable=predator_learning_var, values=["DQN", "PER", "DUELING", "DOUBLE"], width=10)
    predator_learning_menu.grid(row=3, column=3, padx=10, pady=(0, 0), sticky="w")

    
    
    
    
    
    
    # Subtítulo para Design Variant
    design_subtitle = tk.Label(neural_network_frame_models, text="Advanced Layer", font=("Arial", 10, "bold"), fg="black", bg="white")
    design_subtitle.grid(row=4, column=0, columnspan=4, padx=5, pady=(10, 5), sticky="w")

    # Descrição técnica dos Design Variants
    design_description = tk.Label(neural_network_frame_models,
        text=(
            "NONE: Standard network without specialized techniques or enhancements.       \n"
            "RADAR: Uses attention and RGB processing to enhance feature extraction.      "
        ), font=("Verdana", 8), fg="black", bg="white", justify="left", wraplength=480)
    design_description.grid(row=5, column=0, columnspan=4, padx=5, pady=(0, 0), sticky="w")

    # Labels e Comboboxes para Prey e Predator Design Variants
    prey_design_label = tk.Label( neural_network_frame_models, text="Prey:", font=("Verdana", 10), bg="white")
    prey_design_label.grid(row=6, column=0, padx=5, pady=5, sticky="e")

    prey_design_var = tk.StringVar()
    prey_design_var.set("")  # Valor inicial
    prey_design_menu = ttk.Combobox(neural_network_frame_models, textvariable=prey_design_var, values=["NONE", "RADAR"], width=10)
    prey_design_menu.grid(row=6, column=1, padx=10, pady=2, sticky="w")

    predator_design_label = tk.Label( neural_network_frame_models, text="Predator:", font=("Verdana", 10), bg="white")
    predator_design_label.grid(row=6, column=2, padx=5, pady=5, sticky="e")

    predator_design_var = tk.StringVar()
    predator_design_var.set("")  # Valor inicial
    predator_design_menu = ttk.Combobox(neural_network_frame_models, textvariable=predator_design_var, values=["NONE", "RADAR"], width=10)
    predator_design_menu.grid(row=6, column=3, padx=10, pady=5, sticky="w")

 
################################################################################################################################################


    # Criar o frame para os botões e posicioná-lo abaixo da tabela no lado direito da aba Reviews
    button_frame_models = tk.Frame(models_right_frame, bg="white", width=right_frame_width, height=115)
    button_frame_models.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
    button_frame_models.grid_propagate(False)  # Impede o redimensionamento automático do frame

    # Configurar as colunas no frame de botões para a aba Reviews
    button_frame_models.grid_columnconfigure(0, weight=1)  # Espaço à esquerda
    button_frame_models.grid_columnconfigure(1, weight=0)  # Botão Open
    button_frame_models.grid_columnconfigure(2, weight=0)  # Botão Save
    button_frame_models.grid_columnconfigure(3, weight=0)  # Botão Clear
    button_frame_models.grid_columnconfigure(4, weight=1)  # Espaço à direita (ajuste para centralizar)

    # Definindo as imagens como variáveis globais para evitar que o garbage collector as remova
    global open_button_image_m, clear_button_image_m, save_button_image_m

    # Carregar as imagens para a aba Reviews, se disponível
    try:
        
        open_button_image_m = load_image("open_button.png")
        clear_button_image_m = load_image("clear_button.png")
        save_button_image_m = load_image("save_button.png")

    except Exception as e:
        print(f"Error loading image: {e}")

    # Adicionar os botões à aba models, similar à aba Simulation
    if open_button_image_m:
        open_button_models = tk.Button(button_frame_models, image=open_button_image_m, command=open_data_for_model, borderwidth=0, bg="white")
        open_button_models.grid(row=0, column=1, padx=10, pady=10, sticky="e")
    else:
        open_button_models = tk.Button(button_frame_models, text="Open", command=open_data_for_model, bg="white")
        open_button_models.grid(row=0, column=1, padx=10, pady=10, sticky="e")

    # Adicionar o botão 'Clear' com imagem na aba models
    if clear_button_image_m:
        clear_button_models = tk.Button(button_frame_models, image=clear_button_image_m, command=save_data_to_json, borderwidth=0, bg="white")
        clear_button_models.grid(row=0, column=2, padx=10, pady=10, sticky="e")
    else:
        clear_button_models = tk.Button(button_frame_models, text="Clear", command=clear_fields, bg="white")
        clear_button_models.grid(row=0, column=2, padx=10, pady=10, sticky="e")


    if save_button_image_m:
        save_button_models = tk.Button(button_frame_models, image=save_button_image_m, command=save_data_to_json, borderwidth=0, bg="white")
        save_button_models.grid(row=0, column=3, padx=10, pady=10, sticky="e")
    else:
        save_button_models = tk.Button(button_frame_models, text="Save", command=save_data_to_json, bg="white")
        save_button_models.grid(row=0, column=3, padx=10, pady=10, sticky="e")
 

    # =============================================================================================================================================
    # ABA SIMULATION
    # =============================================================================================================================================

    # Definir altura e largura total da janela
    window_width = 900
    window_height = 600

    # Cálculo para definir a largura dos frames proporcionalmente
    left_frame_width = int(window_width * 0.40)  # 40% da largura da janela
    right_frame_width = int(window_width * 0.60)  # 60% da largura da janela

    # Ajustar a altura total, subtraindo o espaço para botões e margens
    frame_height = window_height - 115  # Altura disponível após subtrair o espaço da barra de abas e cabeçalho

    # Configuração das colunas principais no reviews_tab
    simulation_tab.grid_columnconfigure(0, weight=1)  # Coluna esquerda
    simulation_tab.grid_columnconfigure(1, weight=0)  # Separador
    simulation_tab.grid_columnconfigure(2, weight=1)  # Coluna direita

    # Criar o left_frame (40% da largura) com altura ajustada
    simulation_left_frame = tk.Frame(simulation_tab, bg="white", width=left_frame_width, height=frame_height)
    simulation_left_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    simulation_left_frame.grid_columnconfigure(0, weight=1)  # Faz a coluna 0 expandir e se centralizar
    simulation_left_frame.grid_propagate(False)  # Impedir que o frame se redimensione automaticamente

    # Criar o right_frame (60% da largura) com altura ajustada
    simulation_right_frame = tk.Frame(simulation_tab, bg="white", width=right_frame_width, height=frame_height)
    simulation_right_frame.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)
    simulation_right_frame.grid_propagate(False) 

    #######################################################################################
    # Criar um único LabelFrame para todos os campos, com título "Simulation Configuration"
    simulation_label_frame = tk.LabelFrame(simulation_left_frame, text="Simulation Configuration", font=("Arial", 11, "bold"), bg="white", padx=10, pady=0)
    simulation_label_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)

    # Linha separada para Model (Nome do Modelo)
    sim_name_label = tk.Label(simulation_label_frame, text="Model Name:", font=("Verdana", 10, "bold"), bg="white")
    sim_name_label.grid(row=0, column=0, pady=(5, 6), padx=10, sticky="w")

    sim_name_value = tk.Label(simulation_label_frame, text="", font=("Verdana", 10), bg="white")
    sim_name_value.grid(row=0, column=1, pady=(5, 6), padx=10, sticky="e")

    # Separador acima da seção de Grid, Episodes e Steps
    sim_separator_top = tk.Frame(simulation_label_frame, height=2, bd=1, relief=tk.SUNKEN, bg="gray")
    sim_separator_top.grid(row=1, column=0, pady=2, padx=5, sticky="ew", columnspan=4)

    # Configurar as colunas dentro do simulation_label_frame
    simulation_label_frame.grid_columnconfigure(0, weight=0)  # Coluna para os labels (Grid Size, Episodes, Steps)
    simulation_label_frame.grid_columnconfigure(1, weight=1)  # Coluna para os valores

    # Linha separada para Grid Size (Tamanho do Grid)
    sim_grids_label = tk.Label(simulation_label_frame, text="Grid Size:", font=("Verdana", 10, "bold"), bg="white")
    sim_grids_label.grid(row=2, column=0, pady=2, padx=(10, 5), sticky="w")  # Alinhado à esquerda

    sim_grids_value = tk.Label(simulation_label_frame, text="", font=("Verdana", 10), bg="white")
    sim_grids_value.grid(row=2, column=1, pady=2, padx=7, sticky="e")  # Alinhado à direita

    # Linha separada para Episodes e Steps
    sim_episodes_label = tk.Label(simulation_label_frame, text="Episodes:", font=("Verdana", 10, "bold"), bg="white")
    sim_episodes_label.grid(row=3, column=0, pady=2, padx=(10, 5), sticky="w")  # Alinhado à esquerda

    sim_episodes_value = tk.Label(simulation_label_frame, text="", font=("Verdana", 10), bg="white")
    sim_episodes_value.grid(row=3, column=1, pady=2, padx=7, sticky="e")  # Alinhado à direita

    sim_steps_label = tk.Label(simulation_label_frame, text="Steps:", font=("Verdana", 10, "bold"), bg="white")
    sim_steps_label.grid(row=4, column=0, pady=2, padx=(10, 5), sticky="w")  # Alinhado à esquerda

    sim_steps_value = tk.Label(simulation_label_frame, text="", font=("Verdana", 10), bg="white")
    sim_steps_value.grid(row=4, column=1, pady=2, padx=7, sticky="e")  # Alinhado à direita


    # Separador abaixo da seção de Grid, Episodes e Steps
    sim_separator_bottom = tk.Frame(simulation_label_frame, height=2, bd=1, relief=tk.SUNKEN, bg="gray")
    sim_separator_bottom.grid(row=5, column=0, pady=(2, 4), padx=5, sticky="ew", columnspan=4)

    # Tabela de População (Prey e Predator)
    sim_population_frame = tk.Frame(simulation_label_frame, bg="white")
    sim_population_frame.grid(row=6, column=0, columnspan=4, padx=5, pady=(2, 2), sticky="nsew")

    # Configurar as colunas para distribuição uniforme
    sim_population_frame.grid_columnconfigure(0, weight=1)  # Coluna dos labels (títulos)
    sim_population_frame.grid_columnconfigure(1, weight=1)  # Coluna Prey
    sim_population_frame.grid_columnconfigure(2, weight=1)  # Coluna Predator

    # Cabeçalhos Prey e Predator
    sim_prey_label = tk.Label(sim_population_frame, text="Prey", font=("Verdana", 10, "bold"), bg="white", width=7)
    sim_prey_label.grid(row=0, column=1, padx=5, pady=(2, 5), sticky="ew")

    sim_predator_label = tk.Label(sim_population_frame, text="Predator", font=("Verdana", 10, "bold"), bg="white", width=7)
    sim_predator_label.grid(row=0, column=2, padx=5, pady=(2, 5), sticky="ew")

    # Labels e valores de Initial Count (Contagem Inicial)
    sim_inicial_label = tk.Label(sim_population_frame, text="Initial Count", font=("Verdana", 10, "bold"), bg="white")
    sim_inicial_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

    sim_prey_inicial_value = tk.Label(sim_population_frame, text="", font=("Verdana", 10), bg="white")
    sim_prey_inicial_value.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

    sim_predator_inicial_value = tk.Label(sim_population_frame, text="", font=("Verdana", 10), bg="white")
    sim_predator_inicial_value.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

    # Labels e valores de Max Count (Contagem Máxima)
    sim_limite_label = tk.Label(sim_population_frame, text="Max Count", font=("Verdana", 10, "bold"), bg="white")
    sim_limite_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")

    sim_prey_limite_value = tk.Label(sim_population_frame, text="", font=("Verdana", 10), bg="white")
    sim_prey_limite_value.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

    sim_predator_limite_value = tk.Label(sim_population_frame, text="", font=("Verdana", 10), bg="white")
    sim_predator_limite_value.grid(row=2, column=2, padx=5, pady=5, sticky="ew")

    # Labels e valores de Spawn Rate (Taxa de Reprodução)
    sim_reproduction_label = tk.Label(sim_population_frame, text="Spawn Rate", font=("Verdana", 10, "bold"), bg="white")
    sim_reproduction_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")

    sim_prey_reproduction_value = tk.Label(sim_population_frame, text="", font=("Verdana", 10), bg="white")
    sim_prey_reproduction_value.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

    sim_predator_reproduction_value = tk.Label(sim_population_frame, text="", font=("Verdana", 10), bg="white")
    sim_predator_reproduction_value.grid(row=3, column=2, padx=5, pady=5, sticky="ew")

    # Labels e valores de Step Decay (Decaimento por Passo)
    sim_fatigue_label = tk.Label(sim_population_frame, text="Step Decay", font=("Verdana", 10, "bold"), bg="white")
    sim_fatigue_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")

    sim_prey_fatigue_value = tk.Label(sim_population_frame, text="", font=("Verdana", 10), bg="white")
    sim_prey_fatigue_value.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

    sim_predator_fatigue_value = tk.Label(sim_population_frame, text="", font=("Verdana", 10), bg="white")
    sim_predator_fatigue_value.grid(row=4, column=2, padx=5, pady=5, sticky="ew")

    # Labels e valores de Learning Model (Modelo de Aprendizado)
    sim_learning_label = tk.Label(sim_population_frame, text="Learning Model", font=("Verdana", 10, "bold"), bg="white")
    sim_learning_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")

    sim_prey_learning_value = tk.Label(sim_population_frame, text="", font=("Verdana", 10), bg="white")
    sim_prey_learning_value.grid(row=5, column=1, padx=5, pady=5, sticky="ew")

    sim_predator_learning_value = tk.Label(sim_population_frame, text="", font=("Verdana", 10), bg="white")
    sim_predator_learning_value.grid(row=5, column=2, padx=5, pady=5, sticky="ew")

    # Labels e valores de Design Variant (Variante de Design)
    sim_design_label = tk.Label(sim_population_frame, text="Advanced layer", font=("Verdana", 10, "bold"), bg="white")
    sim_design_label.grid(row=6, column=0, padx=5, pady=5, sticky="w")

    sim_prey_design_value = tk.Label(sim_population_frame, text="", font=("Verdana", 10), bg="white")
    sim_prey_design_value.grid(row=6, column=1, padx=5, pady=5, sticky="ew")

    sim_predator_design_value = tk.Label(sim_population_frame, text="", font=("Verdana", 10), bg="white")
    sim_predator_design_value.grid(row=6, column=2, padx=5, pady=5, sticky="ew")

    # Labels e valores de Offline Communication (Comunicação Offline)
    #sim_attention_label = tk.Label(sim_population_frame, text="Communication", font=("Verdana", 10, "bold"), bg="white")
    #sim_attention_label.grid(row=7, column=0, padx=5, pady=5, sticky="w")

    sim_prey_attention_value = tk.Label(sim_population_frame, text="", font=("Verdana", 10), bg="white")
    sim_prey_attention_value.grid(row=7, column=1, padx=5, pady=5, sticky="ew")

    sim_predator_attention_value = tk.Label(sim_population_frame, text="", font=("Verdana", 10), bg="white")
    sim_predator_attention_value.grid(row=7, column=2, padx=5, pady=5, sticky="ew")

    # Criando o right_frame (60% da largura) com altura ajustada
    simulation_right_frame = tk.Frame(simulation_tab, bg="white", width=right_frame_width, height=frame_height)
    simulation_right_frame.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)
    simulation_right_frame.grid_propagate(False)  # Impede que o frame se ajuste automaticamente ao conteúdo

    # Criar o LabelFrame para Machine Learning
    console_frame = tk.LabelFrame(simulation_right_frame, text="Console", font=("Arial", 11, "bold"), bg="white", padx=10, pady=0)
    console_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)

    # Configurar as colunas e linhas dentro do frame do console
    console_frame.grid_columnconfigure(0, weight=1)  # O console (Text) ocupa a coluna 0
    console_frame.grid_columnconfigure(1, weight=0)  # A scrollbar ocupa a coluna 1
    console_frame.grid_rowconfigure(0, weight=1)  # O console (Text) e a scrollbar se expandem verticalmente

    # Criar o widget Text (console) dentro do frame com largura reduzida (ajuste o width conforme necessário)
    console_text = tk.Text(console_frame, bg="black", fg="white", wrap="none", width=40, height=15, insertbackground="white")
    console_text.grid(row=0, column=0, sticky="nsew", pady=(10, 10))  # Expande para ocupar todo o espaço da coluna 0

    # Criar a scrollbar para o console
    scrollbar = tk.Scrollbar(console_frame, orient=tk.VERTICAL, command=console_text.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")  # A scrollbar ocupa a coluna 1

    # Configurar o console para utilizar a scrollbar
    console_text.configure(yscrollcommand=scrollbar.set)

    # Inserir o texto fixo na primeira linha do console
    console_text.config(state=tk.NORMAL)
    console_text.insert("1.0", "press run to start simulation...\n")
    console_text.config(state=tk.DISABLED)

    # Garantir que o frame principal `simulation_right_frame` expanda corretamente
    simulation_right_frame.grid_columnconfigure(0, weight=1)
    simulation_right_frame.grid_rowconfigure(0, weight=1)  # Isso faz o console_frame expandir corretamente

    # Posicionar os botões abaixo do console
    button_frame_simulation = tk.Frame(simulation_right_frame, bg="white", width=right_frame_width, height=115)
    button_frame_simulation.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
    button_frame_simulation.grid_propagate(False)  # Impede o redimensionamento automático do frame


    # Configurar as colunas no frame de botões
    button_frame_simulation.grid_columnconfigure(0, weight=1)  # Espaço à esquerda
    button_frame_simulation.grid_columnconfigure(1, weight=0)  # Botão Open
    button_frame_simulation.grid_columnconfigure(2, weight=0)  # Botão Save
    button_frame_simulation.grid_columnconfigure(3, weight=0)  # Botão Clear
    button_frame_simulation.grid_columnconfigure(4, weight=1)  # Espaço à direita (ajuste para centralizar)

    # Definindo as imagens como variáveis globais para evitar que o garbage collector as remova
    global open_button_image_s, run_button_image_s, cancel_button_image_s, clear_button_image_s, report_button_image_s

    # Carregar as imagens para a aba Reviews, se disponível
    try:
        
        open_button_image_s = load_image("open_button.png")
        run_button_image_s = load_image("run_button.png")
        cancel_button_image_s = load_image("cancel_button.png")
        clear_button_image_s = load_image("clear_button.png")
        report_button_image_s = load_image("report_button.png")

    except Exception as e:
        print(f"Error loading image: {e}")

    # Adicionar o botão 'Open' com imagem e alinhá-lo à direita
    if open_button_image_s:
        open_button_simulation = tk.Button(button_frame_simulation, image=open_button_image_s, command=read_simulation, borderwidth=0, bg="white")
        open_button_simulation.grid(row=0, column=1, padx=10, pady=10, sticky="e")
    else:
        open_button_simulation = tk.Button(button_frame_simulation, text="Open", command=read_simulation, bg="white")
        open_button_simulation.grid(row=0, column=1, padx=10, pady=10, sticky="e")

    # Adicionar o botão 'Clear' com imagem e alinhá-lo à direita
    if clear_button_image_s:
        #clear_button_simulation = tk.Button(button_frame_simulation, image=clear_button_image_s, command=clear_fields_sim, borderwidth=0, bg="white")
        clear_button_simulation = tk.Button(button_frame_simulation, image=clear_button_image_s, 
                                            command=lambda: iniciar_visualizacao(verify_monitor_in_simulation()),
                                            borderwidth=0, bg="white")
         
        clear_button_simulation.grid(row=0, column=2, padx=10, pady=10, sticky="e")
    else:
        clear_button_simulation = tk.Button(button_frame_simulation, text="Clear", command=clear_fields_sim, bg="white")
        clear_button_simulation.grid(row=0, column=2, padx=10, pady=10, sticky="e")

    # Adicionar o botão 'Save' com imagem e alinhá-lo à direita
    if run_button_image_s:
        run_button_simulation = tk.Button(button_frame_simulation, image=run_button_image_s, command=lambda: run_simulate_in_thread(console_text), borderwidth=0, bg="white")
        run_button_simulation.grid(row=0, column=3, padx=10, pady=10, sticky="e")
    else:
        run_button_simulation = tk.Button(button_frame_simulation, text="Save", command=lambda: run_simulate_in_thread(console_text), bg="white")
        run_button_simulation.grid(row=0, column=3, padx=10, pady=10, sticky="e")

    cancel_button_simulation = tk.Button(button_frame_simulation, image=cancel_button_image_s, command=cancel_simulation, borderwidth=0, bg="white")
    cancel_button_simulation.grid_remove()
    

    # =============================================================================================================================================
    # ABA REVIEWS
    # =============================================================================================================================================

    # Configuração das colunas principais no reviews_tab (mesma proporção que a aba Simulation)
    reviews_tab.grid_columnconfigure(0, weight=1)  # Coluna esquerda
    reviews_tab.grid_columnconfigure(1, weight=0)  # Separador
    reviews_tab.grid_columnconfigure(2, weight=1)  # Coluna direita

    # Criar o left_frame (40% da largura) com altura ajustada para a aba Reviews
    reviews_left_frame = tk.Frame(reviews_tab, bg="white", width=left_frame_width, height=frame_height)
    reviews_left_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    reviews_left_frame.grid_columnconfigure(0, weight=1)  # Faz a coluna 0 expandir e se centralizar
    reviews_left_frame.grid_propagate(False)  # Impedir que o frame se redimensione automaticamente

    # Criar o right_frame (60% da largura) com altura ajustada para a aba Reviews
    reviews_right_frame = tk.Frame(reviews_tab, bg="white", width=right_frame_width, height=frame_height)
    reviews_right_frame.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)
    reviews_right_frame.grid_propagate(False)  # Impede que o frame se ajuste automaticamente ao conteúdo

    # Criar um único LabelFrame para todos os campos, com título "Review Configuration"
    review_label_frame = tk.LabelFrame(reviews_left_frame, text="Review Configuration", font=("Arial", 11, "bold"), bg="white", padx=10, pady=5)
    review_label_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    # Linha separada para Model (Nome do Modelo)
    rev_name_label = tk.Label(review_label_frame, text="Model Name:", font=("Verdana", 10, "bold"), bg="white")
    rev_name_label.grid(row=0, column=0, pady=(5, 6), padx=10, sticky="w")

    rev_name_value = tk.Label(review_label_frame, text="", font=("Verdana", 10), bg="white")
    rev_name_value.grid(row=0, column=1, pady=(5, 6), padx=10, sticky="e")


    # Separador acima da seção de Grid, Episodes e Steps
    rev_separator_top = tk.Frame(review_label_frame, height=2, bd=1, relief=tk.SUNKEN, bg="gray")
    rev_separator_top.grid(row=1, column=0, pady=2, padx=5, sticky="ew", columnspan=4)

    # Configurar as colunas dentro do review_label_frame
    review_label_frame.grid_columnconfigure(0, weight=0)  # Coluna para os labels (Grid Size, Episodes, Steps)
    review_label_frame.grid_columnconfigure(1, weight=1)  # Coluna para os valores

    # Linha separada para Grid Size (Tamanho do Grid)
    rev_grids_label = tk.Label(review_label_frame, text="Grid Size:", font=("Verdana", 10, "bold"), bg="white")
    rev_grids_label.grid(row=2, column=0, pady=2, padx=(10, 5), sticky="w")  # Alinhado à esquerda

    rev_grids_value = tk.Label(review_label_frame, text="", font=("Verdana", 10), bg="white")
    rev_grids_value.grid(row=2, column=1, pady=2, padx=7, sticky="e")  # Alinhado à direita

    # Linha separada para Episodes e Steps
    rev_episodes_label = tk.Label(review_label_frame, text="Episodes:", font=("Verdana", 10, "bold"), bg="white")
    rev_episodes_label.grid(row=3, column=0, pady=2, padx=(10, 5), sticky="w")  # Alinhado à esquerda

    rev_episodes_value = tk.Label(review_label_frame, text="", font=("Verdana", 10), bg="white")
    rev_episodes_value.grid(row=3, column=1, pady=2, padx=7, sticky="e")  # Alinhado à direita

    rev_steps_label = tk.Label(review_label_frame, text="Steps:", font=("Verdana", 10, "bold"), bg="white")
    rev_steps_label.grid(row=4, column=0, pady=2, padx=(10, 5), sticky="w")  # Alinhado à esquerda

    rev_steps_value = tk.Label(review_label_frame, text="", font=("Verdana", 10), bg="white")
    rev_steps_value.grid(row=4, column=1, pady=2, padx=7, sticky="e")  # Alinhado à direita

    # Separador abaixo da seção de Grid, Episodes e Steps
    rev_separator_bottom = tk.Frame(review_label_frame, height=2, bd=1, relief=tk.SUNKEN, bg="gray")
    rev_separator_bottom.grid(row=5, column=0, pady=(2, 4), padx=5, sticky="ew", columnspan=4)

    # Tabela de População (Prey e Predator)
    rev_population_frame = tk.Frame(review_label_frame, bg="white")
    rev_population_frame.grid(row=6, column=0, columnspan=4, padx=5, pady=(2, 2), sticky="nsew")

    # Configurar as colunas para distribuição uniforme
    rev_population_frame.grid_columnconfigure(0, weight=1)  # Coluna dos labels (títulos)
    rev_population_frame.grid_columnconfigure(1, weight=1)  # Coluna Prey
    rev_population_frame.grid_columnconfigure(2, weight=1)  # Coluna Predator

    # Cabeçalhos Prey e Predator
    rev_prey_label = tk.Label(rev_population_frame, text="Prey", font=("Verdana", 10, "bold"), bg="white", width=7)
    rev_prey_label.grid(row=0, column=1, padx=5, pady=(2, 5), sticky="ew")

    rev_predator_label = tk.Label(rev_population_frame, text="Predator", font=("Verdana", 10, "bold"), bg="white", width=7)
    rev_predator_label.grid(row=0, column=2, padx=5, pady=(2, 5), sticky="ew")

    # Labels e valores de Initial Count (Contagem Inicial)
    rev_inicial_label = tk.Label(rev_population_frame, text="Initial Count", font=("Verdana", 10, "bold"), bg="white")
    rev_inicial_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

    rev_prey_inicial_value = tk.Label(rev_population_frame, text="", font=("Verdana", 10), bg="white")
    rev_prey_inicial_value.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

    rev_predator_inicial_value = tk.Label(rev_population_frame, text="", font=("Verdana", 10), bg="white")
    rev_predator_inicial_value.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

    # Labels e valores de Max Count (Contagem Máxima)
    rev_limite_label = tk.Label(rev_population_frame, text="Max Count", font=("Verdana", 10, "bold"), bg="white")
    rev_limite_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")

    rev_prey_limite_value = tk.Label(rev_population_frame, text="", font=("Verdana", 10), bg="white")
    rev_prey_limite_value.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

    rev_predator_limite_value = tk.Label(rev_population_frame, text="", font=("Verdana", 10), bg="white")
    rev_predator_limite_value.grid(row=2, column=2, padx=5, pady=5, sticky="ew")

    # Labels e valores de Spawn Rate (Taxa de Reprodução)
    rev_reproduction_label = tk.Label(rev_population_frame, text="Spawn Rate", font=("Verdana", 10, "bold"), bg="white")
    rev_reproduction_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")

    rev_prey_reproduction_value = tk.Label(rev_population_frame, text="", font=("Verdana", 10), bg="white")
    rev_prey_reproduction_value.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

    rev_predator_reproduction_value = tk.Label(rev_population_frame, text="", font=("Verdana", 10), bg="white")
    rev_predator_reproduction_value.grid(row=3, column=2, padx=5, pady=5, sticky="ew")

    # Labels e valores de Step Decay (Decaimento por Passo)
    rev_fatigue_label = tk.Label(rev_population_frame, text="Step Decay", font=("Verdana", 10, "bold"), bg="white")
    rev_fatigue_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")

    rev_prey_fatigue_value = tk.Label(rev_population_frame, text="", font=("Verdana", 10), bg="white")
    rev_prey_fatigue_value.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

    rev_predator_fatigue_value = tk.Label(rev_population_frame, text="", font=("Verdana", 10), bg="white")
    rev_predator_fatigue_value.grid(row=4, column=2, padx=5, pady=5, sticky="ew")

    # Labels e valores de Learning Model (Modelo de Aprendizado)
    rev_learning_label = tk.Label(rev_population_frame, text="Learning Model", font=("Verdana", 10, "bold"), bg="white")
    rev_learning_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")

    rev_prey_learning_value = tk.Label(rev_population_frame, text="", font=("Verdana", 10), bg="white")
    rev_prey_learning_value.grid(row=5, column=1, padx=5, pady=5, sticky="ew")

    rev_predator_learning_value = tk.Label(rev_population_frame, text="", font=("Verdana", 10), bg="white")
    rev_predator_learning_value.grid(row=5, column=2, padx=5, pady=5, sticky="ew")

    # Labels e valores de Design Variant (Variante de Design)
    rev_design_label = tk.Label(rev_population_frame, text="Advanced layer", font=("Verdana", 10, "bold"), bg="white")
    rev_design_label.grid(row=6, column=0, padx=5, pady=5, sticky="w")

    rev_prey_design_value = tk.Label(rev_population_frame, text="", font=("Verdana", 10), bg="white")
    rev_prey_design_value.grid(row=6, column=1, padx=5, pady=5, sticky="ew")

    rev_predator_design_value = tk.Label(rev_population_frame, text="", font=("Verdana", 10), bg="white")
    rev_predator_design_value.grid(row=6, column=2, padx=5, pady=5, sticky="ew")

    # Labels e valores de Offline Communication (Comunicação Offline)
    #rev_attention_label = tk.Label(rev_population_frame, text="Communication", font=("Verdana", 10, "bold"), bg="white")
    #rev_attention_label.grid(row=7, column=0, padx=5, pady=5, sticky="w")

    rev_prey_attention_value = tk.Label(rev_population_frame, text="", font=("Verdana", 10), bg="white")
    rev_prey_attention_value.grid(row=7, column=1, padx=5, pady=5, sticky="ew")

    rev_predator_attention_value = tk.Label(rev_population_frame, text="", font=("Verdana", 10), bg="white")
    rev_predator_attention_value.grid(row=7, column=2, padx=5, pady=5, sticky="ew")


#############################################################################################################

    # Garantir que o frame principal `reviews_right_frame` expanda corretamente
    reviews_right_frame.grid_columnconfigure(0, weight=1)
    reviews_right_frame.grid_rowconfigure(0, weight=1)  # Isso faz o console_frame expandir corretamente

#############################################################################################################

    # Criar o frame que irá encapsular os gráficos da aba Reviews
    graphics_frame_reviews = tk.Frame(reviews_right_frame, bg="white")
    graphics_frame_reviews.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)


    initialize_empty_graphics(graphics_frame_reviews)


#############################################################################################################

    # Criar o frame para os botões e posicioná-lo abaixo da tabela no lado direito da aba Reviews
    button_frame_reviews = tk.Frame(reviews_right_frame, bg="white", width=right_frame_width, height=115)
    button_frame_reviews.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
    button_frame_reviews.grid_propagate(False)  # Impede o redimensionamento automático do frame

    # Configurar as colunas no frame de botões para a aba Reviews
    button_frame_reviews.grid_columnconfigure(0, weight=1)  # Espaço à esquerda
    button_frame_reviews.grid_columnconfigure(1, weight=0)  # Botão Open
    button_frame_reviews.grid_columnconfigure(2, weight=0)  # Botão Save
    button_frame_reviews.grid_columnconfigure(3, weight=0)  # Botão Clear
    button_frame_reviews.grid_columnconfigure(4, weight=1)  # Espaço à direita (ajuste para centralizar)

    # Adicionar os botões à aba Reviews, similar à aba Simulation
    if open_button_image_s:
        open_button_reviews = tk.Button(button_frame_reviews, image=open_button_image_s, command=lambda: load_graph_data_and_refresh_canvas(graphics_frame_reviews), borderwidth=0, bg="white")
        open_button_reviews.grid(row=0, column=1, padx=10, pady=10, sticky="e")

    else:
        open_button_reviews = tk.Button(button_frame_reviews, text="Open", command=lambda: load_graph_data_and_refresh_canvas(graphics_frame_reviews), bg="white")
        open_button_reviews.grid(row=0, column=1, padx=10, pady=10, sticky="e")

    # Adicionar o botão 'Clear' com imagem na aba Reviews
    if clear_button_image_s:
        clear_button_reviews = tk.Button(button_frame_reviews, image=clear_button_image_s, command=lambda: clear_fields_rev(graphics_frame_reviews), borderwidth=0, bg="white")
        clear_button_reviews.grid(row=0, column=2, padx=10, pady=10, sticky="e")
    else:
        clear_button_reviews = tk.Button(button_frame_reviews, text="Clear", command=lambda: clear_fields_rev(graphics_frame_reviews), bg="white")
        clear_button_reviews.grid(row=0, column=2, padx=10, pady=10, sticky="e")

    # Adicionar o botão 'Report' com imagem na aba Reviews
    if run_button_image_s:
        run_button_reviews = tk.Button(button_frame_reviews, image=report_button_image_s, 
                                       command=lambda: on_generate_pdf_button_click(verify_report_in_review()) if verify_report_in_review() else None, 
                                       borderwidth=0, bg="white")
        run_button_reviews.grid(row=0, column=3, padx=10, pady=10, sticky="e")
    else:
        run_button_reviews = tk.Button(button_frame_reviews, text="Report", command=lambda: on_generate_pdf_button_click("mi_15.json"), bg="white")
        run_button_reviews.grid(row=0, column=3, padx=10, pady=10, sticky="e")


# Criando a janela principal com ThemedTk para suportar mais temas
root = ThemedTk(theme="arc")
root.title("P2RIME")

# Carregar a nova imagem que você deseja usar como ícone
icon = load_image("icon.png")

# Definir o novo ícone para a janela principal
root.iconphoto(False, icon)

# Centralizando a janela ao chamar a função
center_window(root, 900, 600)
root.config(bg="white")

# Verificar se todas as imagens estão disponíveis
if not check_images_with_delay():
    root.destroy()


# =============================================================================================================================================
# Tela Apresentação
# =============================================================================================================================================

# Adicionando o logo como imagem em vez do título P2RISMA
header_frame = tk.Frame(root, bg="gray")
header_frame.pack(anchor="w", padx=10, pady=20)

# Carregar a imagem do logo
logo_image = tk.PhotoImage(file="images/logo.png")

# Criar um Label com a imagem do logo
logo_label = tk.Label(header_frame, image=logo_image, bg="white")
logo_label.image = logo_image  # Manter a referência da imagem
logo_label.pack(anchor="w", padx=0, pady=0)

# Criando o frame para as seções (Modeling, Simulation, Analysis)
content_frame = tk.Frame(root, bg="white")
content_frame.pack(pady=10, padx=20)

# Configurando as colunas para centralização
content_frame.grid_columnconfigure(0, weight=1)
content_frame.grid_columnconfigure(1, weight=1)
content_frame.grid_columnconfigure(2, weight=1)

# Adicionando imagens e mantendo as referências (Modeling, Simulation, Analysis)
modeling_image = tk.PhotoImage(file="images/modeling_image.png")
simulation_image = tk.PhotoImage(file="images/simulation_image.png")
analysis_image = tk.PhotoImage(file="images/analysis_image.png")

# Section 1: Modeling
modeling_label = tk.Label(content_frame, image=modeling_image, bg="white")
modeling_label.image = modeling_image  # Manter a referência da imagem
modeling_label.grid(row=0, column=0, padx=5, sticky="nsew")

modeling_title = tk.Label(content_frame, text="1. Model", font=("Arial Black", 12), fg="black", bg="white")
modeling_title.grid(row=1, column=0, sticky="nsew")

modeling_description = tk.Label(content_frame, text="Sets up the essential parameters and variables\nfor the construction and behavior of agents.", 
                                font=("Verdana", 10), fg="black", bg="white", wraplength=200)
modeling_description.grid(row=2, column=0, sticky="nsew")

# Section 2: Simulation
simulation_label = tk.Label(content_frame, image=simulation_image, bg="white")
simulation_label.image = simulation_image  # Manter a referência da imagem
simulation_label.grid(row=0, column=1, padx=20, sticky="nsew")

simulation_title = tk.Label(content_frame, text="2. Simulate", font=("Arial Black", 12), fg="black", bg="white")
simulation_title.grid(row=1, column=1, sticky="nsew")

simulation_description = tk.Label(content_frame, text="Runs the model in real-time to generate\ndata for visualization and analysis.", 
                                  font=("Verdana", 10), fg="black", bg="white", wraplength=200)
simulation_description.grid(row=2, column=1, sticky="nsew")

# Section 3: Analysis
analysis_label = tk.Label(content_frame, image=analysis_image, bg="white")
analysis_label.image = analysis_image  # Manter a referência da imagem
analysis_label.grid(row=0, column=2, padx=20, sticky="nsew")

analysis_title = tk.Label(content_frame, text="3. Analyze", font=("Arial Black", 12), fg="black", bg="white")
analysis_title.grid(row=1, column=2, sticky="nsew")

analysis_description = tk.Label(content_frame, text="Examines the results and enables comparison\nbetween scenarios using graphs and visual displays.", 
                                font=("Verdana", 10), fg="black", bg="white", wraplength=200)
analysis_description.grid(row=2, column=2, sticky="nsew")

# Descrição de Analysis (com largura controlada por wraplength)
analysis_description = tk.Label(content_frame, text="Examines the results and enables comparison\nbetween scenarios using graphs and visual displays.", 
                                font=("Verdana", 10), fg="black", bg="white", wraplength=200)
analysis_description.grid(row=2, column=2)


# Criando o frame para alinhar o "About" e os botões na parte inferior
bottom_frame = tk.Frame(root, bg="white")
bottom_frame.pack(side="bottom", fill="x", padx=20, pady=20)

# Frame do texto About à esquerda
about_frame = tk.Frame(bottom_frame, bg="white")
about_frame.pack(side="left", fill="both", expand=True, anchor="w")  # Preencher a esquerda e expandir

about_title = tk.Label(about_frame, text="About", font=("Arial Black", 16), fg="gray", bg="white")
about_title.pack(anchor="w")

about_text = tk.Label(about_frame, text=(
    "P2RIME is a Predator-Prey Reinforcement Intelligent Model Engine designed "
    "to simulate dynamic models of predator-prey systems, with a focus on data "
    "collection for evaluation and analysis. It enables detailed characterization "
    "of both the environment and agent species providing a precise understanding "
    "of complex interactions and individual behaviors."

), font=("Verdana", 10), fg="black", bg="white", wraplength=600, anchor="w", justify="left")
about_text.pack(anchor="w")

# Frame dos botões à direita
button_frame = tk.Frame(bottom_frame, bg="white")
button_frame.pack(side="right", padx=20, pady=0)

# Criar botões com imagens e manter referências
start_button_image = tk.PhotoImage(file="images/start_button.png").subsample(1, 1)
close_button_image = tk.PhotoImage(file="images/exit_button.png").subsample(1, 1)

start_button = tk.Button(button_frame, image=start_button_image, command=start_simulation, borderwidth=0, bg="white")
start_button.image = start_button_image  # Manter a referência da imagem
start_button.pack(padx=10, pady=10)

close_button = tk.Button(button_frame, image=close_button_image, command=exit_application, borderwidth=0, bg="white")
close_button.image = close_button_image  # Manter a referência da imagem
close_button.pack(padx=10, pady=10)

# Rodando a interface
root.mainloop()