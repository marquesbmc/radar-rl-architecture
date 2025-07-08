import numpy as np
import random
import itertools
import logging
import cv2


class Env:
    def __init__(self, sizeX, sizeY, ray=3):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.ray = ray
        self.agents = []
        self.objects = []
        self.actions = 8  # Número de ações possíveis

    def reset(self):
        # Inicia ou Reinicia o ambiente e posiciona os objetos no jogo.
        
        self.agents = []
        
        # Embaralha a lista de objetos
        random.shuffle(self.objects)

        for agent in self.objects:
            agent.reset(self)
            self.agents.append(agent)

    def add_agent(self, agent):
        self.objects.append(agent)

    def delete(self, agent):
        self.agents.remove(agent)

    def remove_agent(self, agent):
        self.agents.remove(agent)
        self.objects.remove(agent)

    def new_position(self):
        # Cria uma lista de todas as posições possíveis.
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = list(itertools.product(*iterables))

        # Cria uma lista das posições atuais ocupadas pelos agentes.
        current_positions = [(agent.x, agent.y) for agent in self.agents]

        # Filtra as posições possíveis, removendo as ocupadas.
        available_points = [point for point in points if point not in current_positions]

        # Escolhe aleatoriamente uma das posições disponíveis.
        if available_points:
            return random.choice(available_points)
        else:
            # Loga a mensagem indicando que não há posições disponíveis.
            logging.info("Não foi possível encontrar uma nova posição disponível.")
            return None

    def render_env(self):
        a = np.zeros([self.sizeY, self.sizeX, 3])

        for agent in self.agents:
            if agent.x is not None and agent.y is not None:
                a[agent.y, agent.x, :] = agent.channel

        return a

    def _render_agent_matrix_rgb(self, x, y):
        # Renderiza o ambiente para obter a matriz RGB atual
        a = self.render_env()

        # Calcula o tamanho do recorte com base em self.ray
        recorte_tamanho = 2 * self.ray + 1

        # Inicializa o recorte temporário com cor amarela
        recorte_temp = np.ones((recorte_tamanho, recorte_tamanho, 3)) * np.array([1, 1, 0])  # Amarelo

        # Calcula as coordenadas do recorte dentro do ambiente
        inicio_x = x - self.ray
        inicio_y = y - self.ray
        fim_x = inicio_x + recorte_tamanho
        fim_y = inicio_y + recorte_tamanho

        # Calcula os limites de sobreposição entre o recorte e o ambiente
        sobreposicao_inicio_x = max(inicio_x, 0)
        sobreposicao_inicio_y = max(inicio_y, 0)
        sobreposicao_fim_x = min(fim_x, self.sizeX)
        sobreposicao_fim_y = min(fim_y, self.sizeY)

        # Calcula os índices de destino no recorte temporário
        destino_inicio_x = sobreposicao_inicio_x - inicio_x
        destino_inicio_y = sobreposicao_inicio_y - inicio_y
        destino_fim_x = destino_inicio_x + sobreposicao_fim_x - sobreposicao_inicio_x
        destino_fim_y = destino_inicio_y + sobreposicao_fim_y - sobreposicao_inicio_y

        # Copia a sobreposição do ambiente para o recorte temporário
        recorte_temp[destino_inicio_y:destino_fim_y, destino_inicio_x:destino_fim_x] = \
            a[sobreposicao_inicio_y:sobreposicao_fim_y, sobreposicao_inicio_x:sobreposicao_fim_x]

        # Pinta o elemento central do recorte de branco, ajustando a posição baseada em self.ray
        centro = self.ray
        recorte_temp[centro, centro, :] = np.array([1, 1, 1])  # Branco

        return recorte_temp

    def render_agent(self, agent, target_size=(0, 0)):
        original_render = self._render_agent_matrix_rgb(agent.x, agent.y)  # Aqui, render_agent deve ser uma função existente que retorna um numpy.ndarray

        # Verifica se o target_size é (0, 0) e mantém a imagem original
        if target_size == (0, 0):
            return original_render

        # Calcula a razão de aspecto desejada e a atual
        target_ratio = target_size[1] / target_size[0]
        current_ratio = original_render.shape[0] / original_render.shape[1]

        if current_ratio > target_ratio:
            # A imagem é mais alta do que o desejado: redimensiona pela altura
            new_height = target_size[1]
            new_width = int(new_height / current_ratio)
        else:
            # A imagem é mais larga do que o desejado: redimensiona pela largura
            new_width = target_size[0]
            new_height = int(new_width * current_ratio)

        # Redimensiona a imagem
        resized_img = cv2.resize(original_render, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Calcula o padding necessário
        pad_width = (target_size[0] - new_width) // 2
        pad_height = (target_size[1] - new_height) // 2

        # Aplica padding à imagem redimensionada
        padded_img = cv2.copyMakeBorder(resized_img, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return padded_img

    def move_agent(self, agent, action):
        # Obtém referência ao agente.
        penalize = 0.0
        direction = action

        # Calcula as novas coordenadas com base na direção escolhida.
        new_x, new_y = agent.x, agent.y

        # Atualiza a posição com base na direção, seguindo sentido horário.
        if direction == 0:  # Para cima
            new_y -= 1
        elif direction == 1:  # Para cima e direita (diagonal)
            new_x += 1
            new_y -= 1
        elif direction == 2:  # Para direita
            new_x += 1
        elif direction == 3:  # Para baixo e direita (diagonal)
            new_x += 1
            new_y += 1
        elif direction == 4:  # Para baixo
            new_y += 1
        elif direction == 5:  # Para baixo e esquerda (diagonal)
            new_x -= 1
            new_y += 1
        elif direction == 6:  # Para esquerda
            new_x -= 1
        elif direction == 7:  # Para cima e esquerda (diagonal)
            new_x -= 1
            new_y -= 1

        # Verifica se o novo movimento está fora dos limites do ambiente.
        if new_x < 0 or new_x >= self.sizeX or new_y < 0 or new_y >= self.sizeY:
            # Aplica penalidade se o movimento levar o agente para fora do grid.
            return -50.0

        if not self.is_position_empty_and_valid(new_x, new_y):
            a = self.get_agent_at_position(new_x, new_y)

            if agent.breed == 0 and a.breed == 0:
                return -50.0
            if agent.breed == 2:
                return -50.0

            else:
                # Atualiza a posição do agente se o movimento for válido.
                agent.last_x, agent.last_y = agent.x, agent.y
                agent.x, agent.y = new_x, new_y

        else:
            # Atualiza a posição do agente se o movimento for válido.
            agent.last_x, agent.last_y = agent.x, agent.y
            agent.x, agent.y = new_x, new_y

        return penalize

    def is_position_empty_and_valid(self, x, y):
        """
        Verifica se a posição (x, y) está dentro do ambiente e não está ocupada por um agente.

        Parâmetros:
        x (int): A coordenada x da posição a ser verificada.
        y (int): A coordenada y da posição a ser verificada.

        Retorna:
        bool: True se a posição estiver dentro do ambiente e não estiver ocupada, False caso contrário.
        """
        # Verifica se a posição está dentro dos limites do ambiente
        if x < 0 or x >= self.sizeX or y < 0 or y >= self.sizeY:
            return False  # A posição está fora dos limites do ambiente

        # Verifica se a posição está ocupada por algum agente
        for agent in self.agents:
            if agent.x == x and agent.y == y:
                return False  # A posição está ocupada

        return True

    def get_agent_at_position(self, x, y):
        for agent in self.agents:
            if agent.x == x and agent.y == y:
                return agent
        return None

    def population_count(self):
        """Retorna a quantidade de presas e predadores no ambiente."""
        predator_count = 0
        prey_count = 0

        for agent in self.agents:
            if agent.is_alive is True:
                if agent.breed == 0:
                    predator_count += 1
                elif agent.breed == 2:
                    prey_count += 1

        return prey_count, predator_count

    @staticmethod
    def chebyshev_distance(x1, y1, x2, y2):
        distance = max(abs(x2 - x1), abs(y2 - y1))
        return distance