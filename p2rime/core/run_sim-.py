import os
from Agents import Prey, Predator
from Env import Env
from nn_dqn import nn_dqn
from Sim import Sim
import tkinter as tk

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def run_sim(console_output=None):
    def tk_print(*args, **kwargs):
        if console_output:
            console_output.write(' '.join(map(str, args)) + '\n')
            console_output.see(tk.END)  # Rola automaticamente para o final
        else:
            print(*args, **kwargs)


    # Código da simulação aqui
    # Substitua todos os 'print()' por 'tk_print()' para redirecionar a saída
    tk_print("=== Simulation Started ===")

    task_name = "nn_dqn"
    size = 50

    # Configuração do ambiente
    env = Env(sizeX=size, sizeY=size, ray=3)  # Configurações podem variar conforme necessário
    directory = "C:/Users/beLIVE/IA/DECISYS/models/"
    
    num_prey = 50
    num_predator = 25

    num_episodes = 1
    num_steps = 20

    input_shape = (7, 7, 3)

    # Carregar e construir modelos neurais para Prey
    model_nn_prey = nn_dqn()  # Substitua por sua função de criação de modelo
    model_nn_prey.build(input_shape=(None,) + input_shape)
    file_path_model_prey = os.path.join(directory, 'nn_dqn_weights_task1_color_prey.h5')
    model_nn_prey.load_weights(file_path_model_prey)

    # Carregar e construir modelos neurais para Predator
    model_nn_predator = nn_dqn()  # Substitua por sua função de criação de modelo
    model_nn_predator.build(input_shape=(None,) + input_shape)
    file_path_model_predator = os.path.join(directory, 'nn_dqn_weights_task1_color_predator.h5')
    model_nn_predator.load_weights(file_path_model_predator)

    # Povoar o ambiente de teste com agentes
    for j in range(num_prey):  # Número de presas
        pos = env.new_position()
        if pos:
            x, y = pos
            prey = Prey(x, y, env, id=j, nn=model_nn_prey)
            env.add_agent(prey)

    for i in range(num_predator):  # Número de predadores
        pos = env.new_position()
        if pos:
            x, y = pos
            predator = Predator(x, y, env, id=i, nn=model_nn_predator)
            env.add_agent(predator)

    # Executando a sessão de teste
    testing_session = Sim(task_name, env, num_predator, num_prey, model_nn_predator, model_nn_prey, num_episodes, num_steps)
    testing_results = testing_session.run()
    
    tk_print("=== Simulation Ended ===")

    # Exibe o resultado final da simulação
    print(testing_results)


# Executando o teste
if __name__ == "__main__":
    run_sim()
