# **P2RIME - Predator-Prey Reinforcement Intelligent Multiagent Engine**

O **P2RIME** é uma plataforma avançada para modelagem, simulação e análise de interações entre agentes predadores e presas utilizando algoritmos de aprendizado por reforço profundo e técnicas de redes neurais. O sistema foi projetado para suportar a experimentação em ambientes multiagente, fornecendo ferramentas robustas de visualização e geração de relatórios.

---

## **Visão Geral**
O sistema implementa:
- **Ambiente de Simulação:** Modelo dinâmico para interações entre agentes em grids bidimensionais.
- **Redes Neurais:** Modelos de aprendizado profundo para treinar agentes com capacidades adaptativas.
- **Visualização:** Monitoramento em tempo real das simulações com gráficos interativos.
- **Relatórios:** Geração de relatórios quantitativos e qualitativos em formato PDF.

---

## **Estrutura do Projeto**

### **Arquivos Principais**
1. **`p2rime.py`**: Interface principal do sistema que integra o ambiente, a simulação e a interface gráfica.
2. **`monitor.py`**: Modulo para visualização em tempo real das simulações realizadas.
3. **`report.py`**: Ferramentas para análise de dados e geração de relatórios em PDF.

### **Dependências**
O sistema utiliza diversas bibliotecas Python, incluindo:
- **Interface Gráfica**: `tkinter`, `PyQt5`, `ttkthemes`
- **Análise de Dados**: `pandas`, `numpy`
- **Visualização**: `matplotlib`, `reportlab`
- **Machine Learning**: Integração com modelos em `core.nn_dqn`.

---

## **Funcionalidades**

### **1. Aba: Model**

#### **Responsabilidade**
A aba **Model** é responsável por configurar e personalizar os parâmetros iniciais do ambiente e dos agentes antes de iniciar a simulação. Nela, o usuário define as características do grid, agentes e modelos de aprendizado.

#### **Itens e Descrição**
- **Configurações do Ambiente (Environment):**
  - **Grid Size:** Tamanho do grid bidimensional onde a simulação ocorrerá.
  - **Episodes:** Número de episódios a serem simulados.
  - **Steps per Episode:** Número máximo de passos por episódio.

- **Configurações dos Agentes (Population):**
  - **Prey Initial Count:** Número inicial de presas.
  - **Predator Initial Count:** Número inicial de predadores.
  - **Prey Max Count:** Número máximo de presas permitido.
  - **Predator Max Count:** Número máximo de predadores permitido.
  - **Prey Spawn Rate:** Taxa de reprodução das presas (%).
  - **Predator Spawn Rate:** Taxa de reprodução dos predadores (%).

- **Características dos Agentes (Neural Network):**
  - **Learning Model:** Modelos de aprendizado para presas e predadores.
  - **Advanced Layer:** Configuração avançada das redes neurais.
  - **Communication:** Comunicação offline habilitada ou desabilitada.

- **Botões:**
  - **Save:** Salva as configurações em um arquivo JSON.
  - **Clear Fields:** Limpa todos os campos.
  - **Load Model:** Carrega configurações previamente salvas.

---

### **2. Aba: Simulate**

#### **Responsabilidade**
A aba **Simulate** é destinada à execução e controle das simulações baseadas nas configurações definidas na aba **Model**. Ela fornece feedback em tempo real sobre o andamento da simulação.

#### **Itens e Descrição**
- **Parâmetros Carregados:**
  - Todos os parâmetros configurados na aba **Model** são exibidos para validação antes do início da simulação.

- **Console de Saída:**
  - Exibe logs em tempo real da simulação, incluindo eventos como reprodução, morte e progresso por episódio.

- **Botões:**
  - **Run Simulation:** Inicia a simulação.
  - **Cancel Simulation:** Interrompe a simulação.
  - **Clear Fields:** Limpa o console e os campos visuais.

- **Validação:**
  - Verifica a integridade dos parâmetros obrigatórios antes de iniciar a simulação.

---

### **3. Aba: Analyze**

#### **Responsabilidade**
A aba **Analyze** é responsável por visualizar e explorar os resultados das simulações realizadas. Ela oferece gráficos, estatísticas e relatórios detalhados.

#### **Itens e Descrição**
- **Dados Carregados:**
  - Carrega os resultados das simulações salvos em JSON.
  - Exibe as configurações originais e os dados coletados.

- **Gráficos:**
  - **População de Presas:** Evolução do número de presas ao longo dos episódios.
  - **População de Predadores:** Evolução do número de predadores ao longo dos episódios.
  - **Prey vs Predator:** Gráfico de dispersão correlacionando presas e predadores.
  - **População Geral:** Evolução combinada das populações.

- **Estatísticas Quantitativas:**
  - Métricas como média, desvio padrão, mediana, etc., para presas e predadores.

- **Relatório PDF:**
  - Geração de relatório detalhado com gráficos e estatísticas.

- **Botões:**
  - **Load Simulation Data:** Carrega os dados para análise.
  - **Clear Fields:** Limpa todos os campos e gráficos.
  - **Generate Report:** Gera um relatório em PDF.

---

## **Configuração e Execução**

### **Instalação**
1. Clone o repositório:
   ```bash
   git clone https://github.com/usuario/p2rime.git
   cd p2rime
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

### **Execução**
1. Execute o sistema:
   ```bash
   python p2rime.py
   ```
2. Utilize a interface gráfica para configurar, simular e analisar os resultados.

---
