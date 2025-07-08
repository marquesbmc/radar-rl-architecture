import numpy as np
import random
import os
import json
import csv
from core.Agents import Prey, Predator
from tkinter import messagebox


class Sim:
    def __init__(self, task_name, env, num_predators, num_preys, limit_pop_predator, limit_pop_prey, model_nn_predator, model_nn_prey, num_episodes=100, num_steps=10):
        self.directory = f"C:/Users/beLIVE/IA/DECISYS"
        self.model_dir = os.path.join(".", "save", "models")
        os.makedirs(self.model_dir, exist_ok=True)  # Garante que o diretório models existe
        self.task_name = task_name
        self.env = env
        self.num_predators = num_predators
        self.num_preys = num_preys
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.model_nn_predator = model_nn_predator
        self.model_nn_prey = model_nn_prey
        self.count_episodes = 0
        self.limit_pop_predator = limit_pop_predator
        self.limit_pop_prey = limit_pop_prey

        self.m = num_preys
        self.n = num_predators
        self.all_positions = [(x, y) for x in range(self.env.sizeX) for y in range(self.env.sizeY)]

    def check_simulation_data(self, model_data):
        """Verifica se existem dados de simulação no JSON e solicita ao usuário confirmação para excluí-los."""
        if "simulation_data" in model_data and model_data["simulation_data"]:
            if "quantitative_data" in model_data and (model_data["quantitative_data"]["predator"] or model_data["quantitative_data"]["prey"]):
                response = messagebox.askyesno(
                    "Existing Simulation", 
                    "An earlier simulation was detected in this model. Do you want to delete the existing data and start a new simulation?"
                )
                return response
        return True

    def clear_simulation_data(self, model_data):
        """Esvazia os componentes `simulation_data`, `quantitative_data` e `behavior_data` no JSON."""
        model_data["simulation_data"] = []
        model_data["quantitative_data"] = {"predator": [], "prey": []}
        model_data["behavior_data"] = {
            "predator": {"Prey captured": 0, "Nearby prey": 0, "Exploring map": 0},
            "prey": {"Predator escape": 0, "Nearby predator": 0, "Exploring map": 0}
        }

    def update_behavior_data(self, feedback, breed):
        """Verifica se o feedback começa com textos específicos para cada tipo de agente e retorna a chave apropriada."""
        predator_behavior = {
            "Prey captured": "Prey captured",
            "Nearby prey": "Nearby prey",
            "Exploring map": "Exploring map"
        }
        
        prey_behavior = {
            "Predator escape": "Predator escape",
            "Nearby predator": "Nearby predator",
            "Exploring map": "Exploring map"
        }

        # Verifica o breed e se o feedback começa com algum dos textos especificados
        if breed == 0:  # Predador
            for key, start_text in predator_behavior.items():
                if feedback.startswith(start_text):
                    return key
        elif breed == 2:  # Presa
            for key, start_text in prey_behavior.items():
                if feedback.startswith(start_text):
                    return key
        
        return None
    
    def save_agent_sim(self, episode, step, agent, name_sim):
        # Captura o caminho completo do diretório 'save/sim' na pasta atras do arquivo atual
        script_directory = os.path.dirname(os.path.abspath(__file__))  # Diretório do arquivo .py
        parent_directory = os.path.dirname(script_directory)  # Diretório acima do diretório do arquivo
        initial_directory = os.path.join(parent_directory, 'save', 'sim')  # Caminho completo para 'raiz/save/sim'

        # Verifica e cria o diretório 'save/sim' se não existir
        if not os.path.exists(initial_directory):
            os.makedirs(initial_directory)

        # Define o nome do arquivo com base no nome do agente
        filename = os.path.join(initial_directory, f'sim_{name_sim}.csv')
        file_exists = os.path.isfile(filename)
        next_index = 0
        if file_exists:
            with open(filename, mode='r', newline='') as file:
                reader = csv.reader(file)
                next_index = sum(1 for row in reader) - 1  # Subtrai 1 se o cabeçalho existir
        
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(['Index', 'Episode', 'Step', 'Name', 'Channel', 'Position', 'Is Alive'])
            
            # Escreve os dados do agente no arquivo, incluindo o índice.
            writer.writerow([next_index, episode, step, agent.name, agent.channel, (agent.x, agent.y), agent.is_alive])

    def run(self):
        """Executa a simulação e salva os dados diretamente no JSON principal."""
        model_file_path = os.path.join(self.model_dir, f"{self.task_name}.json")

        # Carrega o arquivo modelo
        if os.path.exists(model_file_path):
            with open(model_file_path, 'r') as file:
                model_data = json.load(file)
        else:
            model_data = {}

        # Verifica e pergunta ao usuário se deseja excluir dados de simulação existentes
        if not self.check_simulation_data(model_data):
            print("Simulation canceled by the user.")
            return

        # Limpa os dados de simulação e inicializa `behavior_data`
        self.clear_simulation_data(model_data)
        
        for episode in range(self.num_episodes):
            self.count_episodes = episode
            self.env.reset()

            random.shuffle(self.env.agents)
            episode_rewards = {'predator': 0, 'prey': 0}
            episode_dones = {'predator': 0, 'prey': 0}

            # Transformar todo agent kid em adulto
            for agent in self.env.agents:
                agent.kid = False

            # Reconstrução dos agentes no ambiente
            self.env.objects = []
            self.env.agents = []

            for i in range(self.num_predators):  # Número de predadores
                pos = self.env.new_position()
                if pos:
                    x, y = pos
                    predator = Predator(x, y, self.env, id=i, nn=self.model_nn_predator)
                    self.env.add_agent(predator)

            for j in range(self.num_preys):  # Número de presas
                pos = self.env.new_position()
                if pos:
                    x, y = pos
                    prey = Prey(x, y, self.env, id=j, nn=self.model_nn_prey)
                    self.env.add_agent(prey)
            self.env.reset()

            for step in range(self.num_steps):
                if not any((agent.breed == 2) and agent.is_alive for agent in self.env.agents):
                    break

                if not any((agent.breed == 0) and agent.is_alive for agent in self.env.agents):
                    break

                for agent in self.env.agents:
                    if not agent.is_alive or agent.kid:
                        continue

                    state = self.env.render_agent(agent, (0, 0))
                    state_expanded = np.expand_dims(state, 0)
                    model = self.model_nn_prey if agent.breed == 2 else self.model_nn_predator
                    Q_values = model.predict(state_expanded, verbose=0)
                    action = np.argmax(Q_values[0])
                    _, reward, done, feedback = agent.step(action)


                    # Atualiza os dados de recompensas e terminais
                    if agent.breed == 0:  # Predador
                        episode_rewards['predator'] += reward
                        episode_dones['predator'] += 1 if done else 0
                    elif agent.breed == 2:  # Presa
                        episode_rewards['prey'] += reward
                        episode_dones['prey'] += 1 if done else 0

                    # Dentro do loop de simulação no método run
                    behavior_key = self.update_behavior_data(feedback, agent.breed)
                    if behavior_key:
                        # Incrementa o comportamento correspondente em behavior_data
                        if agent.breed == 2:
                            model_data["behavior_data"]["prey"][behavior_key] += 1
                        else:
                            model_data["behavior_data"]["predator"][behavior_key] += 1

                    #FATIGUE
                    if agent.breed == 2:
                        agent.fatigue += 1
                    if agent.breed == 0 and not done:
                        agent.fatigue += 1
                        
                    if agent.fatigue > agent.fatigue_max:
                        agent.is_alive = False
                        self.env.agents.remove(agent)
                        print(f'MORREU agente fatigado: {agent.name} ')


                    #REPRODUCE
                    pop_predator, pop_prey = self.env.population_count()
                    
                    # REP_PREY
                    if agent.breed == 2 and agent.is_alive and (pop_predator + pop_prey <= self.env.sizeX * self.env.sizeY):
                        
                        num_random = random.randint(0, 99)
                        if num_random < agent.percent_rep and pop_prey <= self.limit_pop_prey:
                          
                            # REP_randomcoord:BEGIN
                            random_position = None
                            # Crie um conjunto com todas as coordenadas ocupadas
                            occupied_positions = [(a.x, a.y) for a in self.env.agents]

                            # Convertendo ambas as listas para sets
                            all_positions_set = set(self.all_positions)
                            occupied_positions_set = set(occupied_positions)

                            # Removendo as posições ocupadas
                            available_positions_set = all_positions_set - occupied_positions_set
                            # Convertendo de volta para uma lista se necessário
                            available_positions = list(available_positions_set)
                            
                            # Inicializa a variável para coordenada vazia
                            coord_empty = False
                            random_position = random.choice(available_positions)

                            if random_position != None:
                                new_x, new_y = random_position
                                coord_empty = True

                            
                            if coord_empty:
                                self.m = self.m + 1
                                new_prey = Prey(new_x, new_y,  self.env, id=self.m, nn=self.model_nn_prey)
                                new_prey.kid = True
                                self.env.agents.append(new_prey)
                                #print(f'Spawn: {agent.name}/{new_prey.name}, ({agent.x}, {agent.y})/({new_prey.x}, {new_prey.y}), target:{num_random}/{agent.percent_rep}')
                            #else:
                                #print(f'{agent.name} não encontrou espaço para reproduzir pois grids: {len(available_positions)}, prop:{ran} !')
                        # REP_randomcoord: END

                    #REP_PREDATOR
                    if agent.breed == 0 and agent.is_alive and (pop_predator + pop_prey <= self.env.sizeX * self.env.sizeY):
                        num_random = random.randint(0, 99)
                        if num_random < agent.percent_rep and pop_predator <= self.limit_pop_predator:

                            # Verifica se a posição está ocupada por algum agente
                            coord_empty = False
                            for a in self.env.agents:
                                if a.x == agent.last_x and a.y == agent.last_y:
                                    # print(f'{agent.name} sem espaço para reproduzir!')
                                    coord_empty = True
                                    break
                            
                            if coord_empty == False:
                                self.n = self.n + 1
                                new_predator = Predator(agent.last_x, agent.last_y,  self.env, id=self.n, nn=self.model_nn_predator)
                                new_predator.kid = True
                                self.env.agents.append(new_predator)
                                #print(f'Spawn: {agent.name}/{new_predator.name}, ({agent.x}, {agent.y})/({new_predator.x}, {new_predator.y}), target:{num_random}/{agent.percent_rep}')
                            # else:
                                # print(f'Predator não reproduziu - prop:{ran} ')
                
                    # PASSOS
                    self.save_agent_sim(episode, step, agent, self.task_name)
                    if agent.breed == 0:
                        if agent.last_hunt:
                            self.save_agent_sim(episode, step, agent.last_hunt, self.task_name)


                    # DONE
                    if done: 
                        agent.is_done = False  # Reset done status
                        if agent.breed == 2: agent.in_danger = False
                        if agent.breed == 0:
                            self.env.agents.remove(agent.current_target)
                            agent.target = None

                    
                    pop_prey, pop_predator = self.env.population_count()
                    result = f"{self.count_episodes}/{step + 1}, {agent.name}: {feedback}"
                    #result = f"{[{self.count_episodes + 1}, {pop_prey}, {pop_predator}]}"
                    yield result

            pop_prey, pop_predator = self.env.population_count()
            model_data["simulation_data"].append([self.count_episodes + 1, pop_prey, pop_predator])

            # Atualiza dados quantitativos para predador e presa
            model_data["quantitative_data"]["predator"].append([round(episode_rewards['predator'], 3), episode_dones['predator'], step])
            model_data["quantitative_data"]["prey"].append([round(episode_rewards['prey'], 3), episode_dones['prey'], step])

        # Salva os dados diretamente no arquivo JSON na pasta `models`
        with open(model_file_path, 'w') as model_file:
            json.dump(model_data, model_file, indent=2)
        
        print(f"Simulation data saved directly to file: {model_file_path}")
        return model_data["quantitative_data"]
