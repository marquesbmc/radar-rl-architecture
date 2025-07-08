# RADAR: Arquitetura Neural Atencional com Fusão de Canais Semânticos

Este repositório contém o código-fonte, os modelos neurais, os notebooks de treinamento, os experimentos simulacionais e os materiais gráficos utilizados na tese de doutorado:

**RADAR: Arquitetura Neural Atencional com Fusão de Canais Semânticos**  
_Uma abordagem para percepção adaptativa e coordenação descentralizada em ambientes multiagente parcialmente observáveis_

---

## 🧠 Sobre a arquitetura RADAR

A arquitetura **RADAR** (Reinforced Attention for Dynamic Agent Relations) foi desenvolvida para enfrentar os desafios perceptivos e estratégicos em ambientes multiagente parcialmente observáveis. Ela integra dois módulos inovadores à estrutura CNN+MLP tradicional:

- **Chromatic Fusion Network (CFN)**: realiza fusão semântica de canais RGB, capturando composições táticas e informações de sobreposição.
- **Spatial Attention Module (SAM)**: aplica atenção espacial híbrida com base em estatísticas locais (pooling médio e norma L2), permitindo adaptação contextual do foco perceptivo.

A arquitetura é avaliada em um ambiente simulado presa–predador com percepção multicanal, obstáculos e movimentação direcional.

---

## 📁 Estrutura do repositório

```
├── fig/         # Figuras da arquitetura, gráficos de simulação e módulos da tese
├── model/       # Modelos treinados (.h5) para agentes predadores e presas
├── notebooks/   # Notebooks para treinamento e execução de simulações
├── p2rime/      # Simulador completo com interface, núcleo e redes neurais
├── sim/         # Diretórios de simulação gerados por diferentes configurações
├── support/     # Scripts auxiliares para visualização e estatísticas
├── train/       # Arquivos de saída de treinamento e análises de desempenho
```

---

## 🚀 Execução

1. Clone o repositório:

```bash
git clone https://github.com/marquesbmc/radar-rl-architecture.git
cd radar-rl-architecture
```

2. Execute os notebooks principais:

- `notebooks/model_training.ipynb`: treinamento dos modelos
- `notebooks/model_simulator.ipynb`: simulações com os modelos salvos

3. Analise os resultados:

```bash
python support/sim_statistics.py
python support/sim_visualization.py
```

Esses scripts processam os dados em `sim_summary.csv` e geram gráficos com as métricas de desempenho.

---

## 📊 Resultados

As comparações entre a arquitetura **RADAR** e a arquitetura clássica **CNN+MLP** foram realizadas utilizando os algoritmos:

- DQN
- Double DQN
- Dueling DQN
- Prioritized Experience Replay (PER)

### As métricas avaliadas incluem:

- Estabilidade no treinamento (*Loss*, *Reward*)
- Evasão/Captura
- Efetividade colaborativa
- Cobertura espacial (exploração)

Os dados e resultados estão disponíveis nos diretórios `sim/` e `support/`.

---

## 🖼️ Figuras incluídas (`fig/`)

- **Arquiteturas neurais:**  
  `arquitetura_radar2.png`, `arquitetura_cnn_mlp.png`, `arquiteturaGeral_radar.png`

- **Módulos:**  
  `module_Chromatic_Fusion_2D.png`, `module_Spatial_Attention.png`

- **Gráficos de simulação e treinamento:**  
  `grafico_treinamento_loss.png`, `grafico_efetividade_colaborativa.png`, `grafico_passos_unicos.png`, `grafico_sucesso_medio.png`

- **Imagens do simulador:**  
  `p2rime_front.png`, `p2rime_simulate.png`, `p2rime_model.png`, `p2rime_analyze.png`

---

## 🧪 Componentes do simulador (`p2rime/`)

- `core/`: ambiente, agentes, simulação
- `neuralnetwork/architecture/`: arquiteturas clássicas e RADAR
- `neuralnetwork/weights/`: pesos dos modelos treinados
- `save/`: simulações, modelos e relatórios PDF
- `images/`: ícones e layout da interface
- `p2rime.py`: interface principal do simulador

---

## 📚 Citação

Se este projeto for útil em sua pesquisa, por favor cite:

**Marques, B. (2025).**  
_RADAR: Arquitetura Neural Atencional com Fusão de Canais Semânticos_.  
Tese de Doutorado, Universidade do Estado do Rio de Janeiro – Programa de Pós-Graduação em Ciências Computacionais e Modelagem Matemática.  
Disponível em: [https://github.com/marquesbmc/radar-rl-architecture](https://github.com/marquesbmc/radar-rl-architecture)

---

## ⚖️ Licença

Este projeto está licenciado sob a **Licença MIT**.  
Consulte o arquivo `LICENSE` para mais detalhes.

---

## 📩 Contato

Para dúvidas ou sugestões, entre em contato por meio do [perfil no GitHub](https://github.com/marquesbmc)  
ou pelo e-mail: **marquesbmc@gmail.com**
