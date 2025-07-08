class Agent:
    def __init__(self, x, y, env, breed, channel):
        self.x = x
        self.y = y
        self.env = env
        self.breed = breed
        self.channel = channel
        self.is_alive = True
        self.is_done = False
        self.current_target = None
        self.current_ally = None
        self.last_x, self.last_y = self.x, self.y
        self.trajectory = 0
        self.fatigue = 0
        self.kid = False
        

        # Category         | Breed | Color  |  Channel   | Color - Att | Channel - Att
        # ---------------------------------------------------------------------------
        # Predator         |   0   | Red    | [1, 0, 0]  | Magenta     | [1, 0, 1]
        # Vegetation       |   1   | Green  | [0, 1, 0]  | --------    | ---------
        # Prey             |   2   | Blue   | [0, 0, 1]  | Cyan        | [0, 1, 1]
        # Outside Grid     |   -   | Yellow | [1, 1, 0]  | --------    | ---------
        # Empty Grid Cell  |   -   | Black  | [0, 0, 0]  | --------    | ---------
        # Reserved Training|   -   | White  | [1, 1, 1]  | --------    | ---------

    def reset(self, env):
        self.x, self.y = env.new_position()
        self.is_alive = True
        self.target = None
        self.is_done = False
        self.trajectory = 0
        self.last_x, self.last_y = self.x, self.y
        self.fatigue = 0
        self.kid = False

    def older(self):
        self.kid = True


    def step(self, action):
        penalty = self.env.move_agent(self, action)
        done, reward, ler = self.check_goal()
        state = self.env.render_agent(self)

        return state, reward + penalty, done, ler


class Prey(Agent):
    def __init__(self, x, y, env, id, fatigue_max = 1000, percent_rep = 10, nn = None, is_on = True):
        super().__init__(x, y, env, breed=2, channel=[0, 0, 1])
        self.name = f"Prey_{id}"
        self.in_danger = False
        self.is_on = is_on
        self.fatigue_max = fatigue_max
        self.percent_rep = percent_rep

        # Inicializa o modelo neural específico para Prey
        self.model_nn = nn
        # DQN(num_actions=8) self.model_nn.build((None,) + (56, 56, 3))


    def check_goal(self):
        done = False
        reward = -0.5  # Recompensa padrão para incentivar a exploração segura.
        feedback = ""
        target_detected = False
        ally_detected = False

        # Corrige a verificação de 'alive' para 'current_target'
        if not hasattr(self, 'current_target') or (self.current_target and not self.current_target.is_alive):
            self.current_target = None  # Reseta o alvo atual se não houver nenhum ou se o alvo não estiver mais vivo.

        # Corrige a verificação de 'alive' para 'current_ally'
        if not hasattr(self, 'current_ally') or (self.current_ally and not self.current_ally.is_alive):
            self.current_ally = None  # Reseta o amigo atual se não houver nenhum ou se o alvo não estiver mais vivo.

        closest_target_distance = float('inf')
        closest_ally_distance = float('inf')

        # Procura por agente vivo, presa ou predador, mais próxima ou mantém o foco na atual.
        for agent in self.env.agents:
            # verifica se é predator
            if agent.breed == 0 and agent.is_alive:  # Checa se o agente é um predador
                distance = self.env.chebyshev_distance(self.x, self.y, agent.x, agent.y)
                if distance <= self.env.ray and (self.current_target is None or distance < closest_target_distance):  # Verifica se o predador está dentro do raio de percepção
                    closest_target_distance = distance
                    self.current_target = agent
                    target_detected = True

            # verifica se é presa com informacao de predator
            elif agent.breed == 2 and agent.is_alive and agent.channel == [0, 1, 1]:  # Se o agente for uma presa com alvo
                distance = self.env.chebyshev_distance(self.x, self.y, agent.x, agent.y)
                if distance <= self.env.ray and (self.current_ally is None or distance < closest_ally_distance):
                    closest_ally_distance = distance
                    self.current_ally = agent
                    ally_detected = True


        #Critérios
        # Verificação de `in_danger`
        if self.in_danger and 0 < closest_ally_distance > 3:
            done = True
            self.in_danger = False
            reward = 50.0
            feedback = f"Predator escape"

        elif target_detected and 0 < closest_target_distance <= 3:
            self.in_danger = True  # Marca que a presa está em perigo.
            reward = -6.0 / closest_target_distance
            feedback = f"Nearby predator: {closest_target_distance}."

        elif ally_detected and 0 < closest_ally_distance <= 3:
            reward = 1.0 / closest_ally_distance
            feedback += f"Proximidade aliado: {closest_ally_distance}."
        else:
            feedback += "Exploring map"

        # Atualiza channel
        for agent in self.env.agents:
            if agent.is_alive and agent.breed == 0:  # Se o agente for um predador vivo
                distance = self.env.chebyshev_distance(self.x, self.y, agent.x, agent.y)
                if distance <= self.env.ray:
                    self.channel = [0, 1, 1]
                    break
                else:
                    self.channel = [0, 0, 1]

        return done, reward, feedback

class Predator(Agent):
    def __init__(self, x, y, env, id, fatigue_max=1000, percent_rep=10, nn=None, is_on=True):
        super().__init__(x, y, env, breed=0, channel=[1, 0, 0])
        self.name = f"Predador_{id}"
        self.last_hunt = None
        self.is_on = is_on
        self.fatigue_max = fatigue_max
        self.percent_rep = percent_rep

        # Inicializa o modelo neural específico para Predator
        self.model_nn = nn


    def check_goal(self):
        done = False
        reward = -0.5  # Recompensa padrão para incentivar a movimentação.
        feedback = ""
        target_detected = False
        ally_detected = False
        self.current_target = None  # Reseta o alvo após a captura.



        # Corrige a verificação de 'alive' para 'current_target'
        if not hasattr(self, 'current_target') or (self.current_target and not self.current_target.is_alive):
            self.current_target = None  # Reseta o alvo atual se não houver nenhum ou se o alvo não estiver mais vivo.

        # Corrige a verificação de 'alive' para 'current_ally'
        if not hasattr(self, 'current_ally') or (self.current_ally and not self.current_ally.is_alive):
            self.current_ally = None  # Reseta o amigo atual se não houver nenhum ou se o alvo não estiver mais vivo.

        closest_target_distance = float('inf')
        closest_ally_distance = float('inf')

        # Procura por agente vivo, presa ou predador, mais próxima ou mantém o foco na atual.
        for agent in self.env.agents:
            # verifica se é presa
            if agent.is_alive and agent.breed == 2:  # Se o agente for uma presa viva
                distance = self.env.chebyshev_distance(self.x, self.y, agent.x, agent.y)
                if distance <= self.env.ray and (self.current_target is None or distance < closest_target_distance):
                    closest_target_distance = distance
                    self.current_target = agent
                    target_detected = True
            
            # verifica se é predador com informacao de presa
            elif agent.breed == 0 and agent.is_alive and agent.channel == [1, 0, 1]:  # Se o agente for um predator com alvo
                distance = self.env.chebyshev_distance(self.x, self.y, agent.x, agent.y)
                if distance <= self.env.ray and (self.current_ally is None or distance < closest_ally_distance):
                    closest_ally_distance = distance
                    self.current_ally = agent
                    ally_detected = True



        #Criterios
        if target_detected and 0 == closest_target_distance:
            done = True
            reward = 50.0  # Recompensa significativa por capturar a presa.
            self.current_target.is_alive = False
            #self.last_hunt = self.current_target # necessário para atualizar o sim.py
            #self.env.agents.remove(self.current_target)  # Remove a presa capturada do ambiente.
            feedback = f"Prey captured: {self.current_target.name}."
            #self.current_target = None  # Reseta o alvo após a captura.
        
        elif target_detected and 0 < closest_target_distance <= 3:           
            reward = 6.0 / (closest_target_distance)  # Recompensa por proximidade com a presa.
            feedback = f"Nearby prey: {closest_target_distance}."

        elif ally_detected and 0 < closest_ally_distance <= 3:
            reward = 0.50 / (closest_ally_distance)  # Recompensa por proximidade com a presa.
            feedback = f"Proximidade aliado: {closest_ally_distance}."
        
        else:
            feedback += "Exploring map"

        #Atualiza channel
        for agent in self.env.agents:
            if agent.is_alive and agent.breed == 2: # Se o agente for uma presa viva
                distance = self.env.chebyshev_distance(self.x, self.y, agent.x, agent.y)
                if distance <= self.env.ray:
                    self.channel = [1, 0, 1]
                    break
                else:
                    self.channel = [1, 0, 0]

        return done, reward, feedback
