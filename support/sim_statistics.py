import pandas as pd
import os
import argparse
from scipy.stats import hmean
import re
import csv

def determine_category(file_name):
    """
    Determina a categoria com base no nome do arquivo.

    Parâmetros:
        file_name (str): Nome do arquivo.

    Retorna:
        str: Categoria identificada.
    """
    if "sz10_s10_py10_pd5_o0.1" in file_name:
        return "padrao"
    elif "sz20_s10_py15_pd10_o0.05" in file_name:
        return "extensivo_escasso"
    elif "sz10_s10_py15_pd10_o0.3" in file_name:
        return "denso_restrito"
    else:
        return ""

def determine_algorithm(file_name):
    """
    Determina o algoritmo com base no nome do arquivo.

    Parâmetros:
        file_name (str): Nome do arquivo.

    Retorna:
        str: Algoritmo identificado.
    """
    if "dueling" in file_name:
        return "dueling"
    elif "per" in file_name:
        return "per"
    elif "double" in file_name:
        return "double"
    else:
        return "dqn"

def determine_type(file_name):
    """
    Determina o tipo com base no nome do arquivo.

    Parâmetros:
        file_name (str): Nome do arquivo.

    Retorna:
        str: Tipo identificado.
    """
    if "radar" in file_name:
        return "radar"
    else:
        return "tradicional"

def load_and_split_data_old(file_path):
    """
    Carrega os dados do arquivo e separa em DataFrames para presas e predadores.

    Parâmetros:
        file_path (str): Caminho para o arquivo de dados.

    Retorna:
        tuple: Dois DataFrames, um para presas e outro para predadores.
    """
    import csv
    import pandas as pd

    # Inicializar listas para armazenar os dados
    data = {
        'episode': [],
        'step': [],
        'breed': [],
        'feedback': [],
        'reward': [],
        'done': [],
        'coordenada': [],  # Para armazenar as coordenadas extraídas
    }

    # Ler o arquivo linha por linha
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            log = row[0]  # Cada linha contém a string completa

            # Extrair os campos manualmente
            try:
                episode = log.split("/")[0]
                step = int(log.split(":")[0].split("/")[1])
                breed = int(log.split("Breed:")[1].split(",")[0])
                feedback = log.split("Feedback: ")[1] if "Feedback: " in log else None
                reward = float(log.split("Reward: ")[1].split(",")[0])
                done = log.split("Done: ")[1].split(",")[0].strip().lower() == 'true'

                # Tentar extrair coordenadas do log (padrão como "(x, y)")
                coordenada = None
                if "Position: " in log:
                    match = re.search(r'Position: \((\d+,\d+)\)', log)
                    if match:
                        coordenada = match.group(1)

                # Adicionar os campos extraídos ao dicionário
                data['episode'].append(episode)
                data['step'].append(step)
                data['breed'].append(breed)
                data['feedback'].append(feedback)
                data['reward'].append(reward)
                data['done'].append(done)
                data['coordenada'].append(coordenada)
            except (IndexError, ValueError) as e:
                raise ValueError(f"Erro ao processar a linha: {log}\nErro: {e}")

    # Converter o dicionário em um DataFrame
    df = pd.DataFrame(data)

    # Dividir em presas e predadores
    prey_data = df[df['breed'] == 2].copy()
    predator_data = df[df['breed'] == 0].copy()

    return prey_data, predator_data

def load_and_split_data(file_path):
    """
    Carrega os dados do arquivo e separa em DataFrames para presas e predadores.

    Parâmetros:
        file_path (str): Caminho para o arquivo de dados.

    Retorna:
        tuple: Dois DataFrames, um para presas e outro para predadores.
    """
    # Inicializar listas para armazenar os dados
    data = {
        'episode': [],
        'step': [],
        'breed': [],
        'feedback': [],
        'reward': [],
        'done': [],
        'coordenada': [],  # Para armazenar as coordenadas extraídas
    }

    # Ler o arquivo linha por linha
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            log = row[0]  # Cada linha contém a string completa

            # Extrair os campos manualmente
            try:
                episode = log.split("/")[0]
                step = int(log.split(":")[0].split("/")[1])
                breed_match = re.search(r'Breed:(\d+)', log)
                breed = int(breed_match.group(1)) if breed_match else None
                feedback = log.split("Feedback: ")[1] if "Feedback: " in log else None
                reward_match = re.search(r'Reward: ([\d\.\-]+)', log)
                reward = float(reward_match.group(1)) if reward_match else None
                done_match = re.search(r'Done: (True|False)', log)
                done = done_match.group(1).lower() == 'true' if done_match else None

                # Tentar extrair coordenadas do log (padrão como "(x, y)")
                coordenada = None
                match = re.search(r'Position: \((\d+), (\d+)\)', log)
                if match:
                    coordenada = f"({match.group(1)}, {match.group(2)})"

                # Adicionar os campos extraídos ao dicionário
                data['episode'].append(episode)
                data['step'].append(step)
                data['breed'].append(breed)
                data['feedback'].append(feedback)
                data['reward'].append(reward)
                data['done'].append(done)
                data['coordenada'].append(coordenada)

            except (IndexError, ValueError, AttributeError) as e:
                raise ValueError(f"Erro ao processar a linha: {log}\nErro: {e}")

    # Converter o dicionário em um DataFrame
    df = pd.DataFrame(data)

    # Dividir em presas e predadores
    prey_data = df[df['breed'] == 2].copy()
    predator_data = df[df['breed'] == 0].copy()

    return prey_data, predator_data

def calcular_total_sucesso_escapes(df):
    """
    Calcula o total de feedbacks de sucesso para "[PREY]: Evasao do alvo" no DataFrame.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados das presas.

    Retorna:
        int: Total de feedbacks de sucesso contendo "[PREY]: Evasao do alvo".
    """
    filtro = df['feedback'].str.contains(r'\[PREY\]: Evasao do alvo', case=False, na=False)
    total_escapes = filtro.sum()
    return total_escapes

def calcular_total_sucesso_capturas(df):
    """
    Calcula o total de feedbacks de sucesso para "[PREDATOR]: Alvo capturado" no DataFrame.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados dos predadores.

    Retorna:
        int: Total de feedbacks de sucesso contendo "[PREDATOR]: Alvo capturado".
    """
    filtro = df['feedback'].str.contains(r'\[PREDATOR\]: Alvo capturado', case=False, na=False)
    total_capturas = filtro.sum()
    return total_capturas

def calcular_media_sucesso_escapes(df):
    filtro = df['feedback'].str.contains(r'\[PREY\]: Evasao do alvo', case=False, na=False)
    df_filtrado = df[filtro]

    # Agrupar por episódio e calcular a média de sucessos por etapa
    #media_por_episodio = df_filtrado.groupby('episode').size() / df.groupby('episode')['step'].max()
    media_por_episodio = df_filtrado.groupby('episode').size()

    # Calcular a média geral dos episódios
    media_geral = media_por_episodio.mean()
    return round(media_geral, 3)

def calcular_media_sucesso_capturas(df):
    filtro = df['feedback'].str.contains(r'\[PREDATOR\]: Alvo capturado', case=False, na=False)
    df_filtrado = df[filtro]

    # Agrupar por episódio e calcular a média de sucessos por etapa
    #media_por_episodio = df_filtrado.groupby('episode').size() / df.groupby('episode')['step'].max()
    media_por_episodio = df_filtrado.groupby('episode').size()

    # Calcular a média geral dos episódios
    media_geral = media_por_episodio.mean()
    return round(media_geral, 3)

def calcular_mediana_sucesso_escapes(df):
    """
    Calcula a mediana de sucessos por etapa para "[PREY]: Evasao do alvo" em cada episódio e,
    em seguida, calcula a mediana dos episódios.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados das presas.

    Retorna:
        float: Mediana de sucessos por etapa para "[PREY]: Evasao do alvo".
    """
    filtro = df['feedback'].str.contains(r'\[PREY\]: Evasao do alvo', case=False, na=False)
    df_filtrado = df[filtro]

    # Agrupar por episódio e calcular a proporção de sucessos por etapa
    proporcao_por_episodio = df_filtrado.groupby('episode').size() / df.groupby('episode')['step'].max()

    # Calcular a mediana dos episódios
    mediana_geral = proporcao_por_episodio.median()
    return round(mediana_geral, 3)

def calcular_mediana_sucesso_capturas(df):
    """
    Calcula a mediana de sucessos por etapa para "[PREDATOR]: Alvo capturado" em cada episódio e,
    em seguida, calcula a mediana dos episódios.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados dos predadores.

    Retorna:
        float: Mediana de sucessos por etapa para "[PREDATOR]: Alvo capturado".
    """
    filtro = df['feedback'].str.contains(r'\[PREDATOR\]: Alvo capturado', case=False, na=False)
    df_filtrado = df[filtro]

    # Agrupar por episódio e calcular a proporção de sucessos por etapa
    proporcao_por_episodio = df_filtrado.groupby('episode').size() / df.groupby('episode')['step'].max()

    # Calcular a mediana dos episódios
    mediana_geral = proporcao_por_episodio.median()
    return round(mediana_geral, 3)

def calcular_desvio_padrao_sucesso_escapes(df):
    """
    Calcula o desvio padrão de sucessos por etapa para "[PREY]: Evasao do alvo" em cada episódio e,
    em seguida, calcula o desvio padrão dos episódios.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados das presas.

    Retorna:
        float: Desvio padrão de sucessos por etapa para "[PREY]: Evasao do alvo".
    """
    filtro = df['feedback'].str.contains(r'\[PREY\]: Evasao do alvo', case=False, na=False)
    df_filtrado = df[filtro]

    # Agrupar por episódio e calcular a proporção de sucessos por etapa
    proporcao_por_episodio = df_filtrado.groupby('episode').size() / df.groupby('episode')['step'].max()

    # Calcular o desvio padrão dos episódios
    desvio_padrao_geral = proporcao_por_episodio.std()
    return round(desvio_padrao_geral, 3)

def calcular_desvio_padrao_sucesso_capturas(df):
    """
    Calcula o desvio padrão de sucessos por etapa para "[PREDATOR]: Alvo capturado" em cada episódio e,
    em seguida, calcula o desvio padrão dos episódios.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados dos predadores.

    Retorna:
        float: Desvio padrão de sucessos por etapa para "[PREDATOR]: Alvo capturado".
    """
    filtro = df['feedback'].str.contains(r'\[PREDATOR\]: Alvo capturado', case=False, na=False)
    df_filtrado = df[filtro]

    # Agrupar por episódio e calcular a proporção de sucessos por etapa
    proporcao_por_episodio = df_filtrado.groupby('episode').size() / df.groupby('episode')['step'].max()

    # Calcular o desvio padrão dos episódios
    desvio_padrao_geral = proporcao_por_episodio.std()
    return round(desvio_padrao_geral, 3)

def calcular_media_harmonica_sucesso_escapes(df):
    """
    Calcula a média harmônica de sucessos por etapa para "[PREY]: Evasao do alvo" em cada episódio e,
    em seguida, calcula a média harmônica dos episódios.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados das presas.

    Retorna:
        float: Média harmônica de sucessos por etapa para "[PREY]: Evasao do alvo".
    """
    filtro = df['feedback'].str.contains(r'\[PREY\]: Evasao do alvo', case=False, na=False)
    df_filtrado = df[filtro]

    # Agrupar por episódio e calcular a proporção de sucessos por etapa
    proporcao_por_episodio = df_filtrado.groupby('episode').size() / df.groupby('episode')['step'].max()

    # Filtrar valores positivos (média harmônica só aceita valores positivos)
    proporcao_valida = proporcao_por_episodio[proporcao_por_episodio > 0]

    # Calcular a média harmônica dos episódios
    if not proporcao_valida.empty:
        media_harmonica_geral = hmean(proporcao_valida)
    else:
        media_harmonica_geral = 0

    return round(media_harmonica_geral, 3)

def calcular_media_harmonica_sucesso_capturas(df):
    """
    Calcula a média harmônica de sucessos por etapa para "[PREDATOR]: Alvo capturado" em cada episódio e,
    em seguida, calcula a média harmônica dos episódios.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados dos predadores.

    Retorna:
        float: Média harmônica de sucessos por etapa para "[PREDATOR]: Alvo capturado".
    """
    filtro = df['feedback'].str.contains(r'\[PREDATOR\]: Alvo capturado', case=False, na=False)
    df_filtrado = df[filtro]

    # Agrupar por episódio e calcular a proporção de sucessos por etapa
    proporcao_por_episodio = df_filtrado.groupby('episode').size() / df.groupby('episode')['step'].max()

    # Filtrar valores positivos (média harmônica só aceita valores positivos)
    proporcao_valida = proporcao_por_episodio[proporcao_por_episodio > 0]

    # Calcular a média harmônica dos episódios
    if not proporcao_valida.empty:
        media_harmonica_geral = hmean(proporcao_valida)
    else:
        media_harmonica_geral = 0

    return round(media_harmonica_geral, 3)

def calcular_media_proximidade_aliados_presas(df):
    """
    Calcula a média de proximidade de aliados para presas por etapa, considerando diferentes níveis de proximidade.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados das presas.

    Retorna:
        float: Média geral de proximidade de aliados para as presas.
    """
    medias_por_proximidade = [
        df[df['feedback'].str.contains(rf'\[PREY\]: Proximidade aliado: {i}', na=False)]
        .groupby(['episode', 'step'])
        .size()
        .groupby('episode')
        .mean()
        .mean()
        for i in range(1, 4)
    ]

    # Calcular a média geral das proximidades
    media_geral = sum(medias_por_proximidade) / len(medias_por_proximidade)
    return round(media_geral, 3)

def calcular_media_proximidade_aliados_predadores(df):
    """
    Calcula a média de proximidade de aliados para predadores por etapa, considerando diferentes níveis de proximidade.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados dos predadores.

    Retorna:
        float: Média geral de proximidade de aliados para os predadores.
    """
    medias_por_proximidade = [
        df[df['feedback'].str.contains(rf'\[PREDATOR\]: Proximidade aliado: {i}', na=False)]
        .groupby(['episode', 'step'])
        .size()
        .groupby('episode')
        .mean()
        .mean()
        for i in range(1, 4)
    ]

    # Calcular a média geral das proximidades
    media_geral = sum(medias_por_proximidade) / len(medias_por_proximidade)
    return round(media_geral, 3)

def calcular_media_coordenadas_unicas(df):
    """
    Calcula a média de coordenadas únicas visitadas por episódio.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados.

    Retorna:
        float: Média de coordenadas únicas visitadas por episódio.
    """
    if df.empty:
        return 0

    # Validar que a coluna 'coordenada' foi preenchida
    if 'coordenada' not in df.columns or df['coordenada'].isnull().all():
        raise ValueError("A coluna 'coordenada' está vazia ou não foi preenchida corretamente.")

    # Obter coordenadas únicas por episódio
    coordenadas_unicas_por_episodio = df.groupby('episode')['coordenada'].apply(lambda x: len(set(x.dropna())))

    # Calcular a média das coordenadas únicas por episódio
    media_coordenadas = coordenadas_unicas_por_episodio.mean()
    return round(media_coordenadas, 3)


def contar_total_coordenadas(df, coluna_coordenadas="coordenada", por_episodio=True):
    """
    Conta o total de coordenadas visitadas.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados das simulações.
        coluna_coordenadas (str): Nome da coluna onde as coordenadas estão armazenadas.
        por_episodio (bool): Se True, retorna o total de coordenadas únicas por episódio.

    Retorna:
        int: Total de coordenadas visitadas.
    """
    if coluna_coordenadas not in df.columns:
        raise ValueError(f"A coluna '{coluna_coordenadas}' não está no DataFrame.")

    # Garantir que a coluna de coordenadas não tenha valores nulos
    df = df.dropna(subset=[coluna_coordenadas])

    if por_episodio:
        coordenadas_por_episodio = (
            df.groupby('episode')[coluna_coordenadas]
            .apply(lambda x: len(x))  # Total de coordenadas visitadas
            .sum()
        )
        return coordenadas_por_episodio
    else:
        total_coordenadas = len(df[coluna_coordenadas])  # Soma de todas as coordenadas visitadas
        return total_coordenadas



def process_files(file_paths):
    """
    Processa uma lista de arquivos e calcula estatísticas relevantes.

    Parâmetros:
        file_paths (list): Lista de caminhos para os arquivos.

    Retorna:
        pd.DataFrame: DataFrame com as estatísticas calculadas.
    """
    summary_rows = []

    for file_path in file_paths:
        prey_data, predator_data = load_and_split_data(file_path)

        # total de sucesso por funcao breed e feedback
        prey_total = calcular_total_sucesso_escapes(prey_data)
        predator_total = calcular_total_sucesso_capturas(predator_data)

        
        # media de sucesso por done
        prey_mean_done = calcular_media_sucesso_escapes(prey_data)
        predator_mean_done = calcular_media_sucesso_capturas(predator_data) 

        # media harmonica de sucesso por done
        prey_harm_done = calcular_media_harmonica_sucesso_escapes(prey_data)
        predator_harm_done = calcular_media_harmonica_sucesso_capturas(predator_data) 

        # mediana de sucesso por done
        prey_mediana_done = calcular_mediana_sucesso_escapes(prey_data)
        predator_mediana_done = calcular_mediana_sucesso_capturas(predator_data)

        # desvio padrao de sucesso por done
        prey_dvp_done = calcular_desvio_padrao_sucesso_escapes(prey_data)
        predator_dvp_done = calcular_desvio_padrao_sucesso_capturas(predator_data)

        # media aliado proximo
        prey_media_proximidade_aliados = calcular_media_proximidade_aliados_presas(prey_data)
        predator_media_proximidade_aliados = calcular_media_proximidade_aliados_predadores(predator_data)

        # media coordenadas unicas
        prey_media_coord_unicas = calcular_media_coordenadas_unicas(prey_data)
        predadores_media_coord_unicas = calcular_media_coordenadas_unicas(predator_data)

        prey_media_coord = contar_total_coordenadas(prey_data)
        predadores_media_coord = contar_total_coordenadas(predator_data)

        summary_row = {
            'file_name': os.path.basename(file_path),
            'category': determine_category(os.path.basename(file_path)),
            'algorithm': determine_algorithm(os.path.basename(file_path)),
            'type': determine_type(os.path.basename(file_path)),

            '[PREY]: total de done': prey_total,
            '[PREDATOR]: total de done': predator_total,
            '[SOMA]: total de done': predator_total + prey_total,
            '[PREY]: media escapes done': prey_mean_done,
            '[PREDATOR]: media captures done': predator_mean_done,
            '[SOMA]: media de done': predator_mean_done + prey_mean_done,
            '[PREY]: harmonica escapes done': prey_harm_done,
            '[PREDATOR]: harmonica captures done': predator_harm_done,
            '[PREY]: mediana escapes done': prey_mediana_done,
            '[PREDATOR]: mediana captures done': predator_mediana_done,
            '[PREY]: dvp escapes done': prey_dvp_done,
            '[PREDATOR]: dvp captures done': predator_dvp_done,
            '[PREY]: media proximidade aliados': prey_media_proximidade_aliados,
            '[PREDATOR]: media proximidade aliados': predator_media_proximidade_aliados,
            '[SOMA]: media proximidade aliados': prey_media_proximidade_aliados + predator_media_proximidade_aliados,
            '[PREY]: media efetividade aliados': round(prey_mean_done / prey_media_proximidade_aliados, 3),
            '[PREDATOR]: media efetividade aliados': round(predator_mean_done / predator_media_proximidade_aliados, 3),
            '[SOMA]: media efetividade aliados': round(predator_mean_done / predator_media_proximidade_aliados, 3) + round(prey_mean_done / prey_media_proximidade_aliados, 3),
            '[PREY]: media coordenadas unicas': prey_media_coord_unicas,
            '[PREDATOR]: media coordenadas unicas': predadores_media_coord_unicas,
            '[SOMA]: media coordenadas unicas': predadores_media_coord_unicas + prey_media_coord_unicas,
            '[PREY]: media coordenadas': prey_media_coord,
            '[PREDATOR]: media coordenadas': predadores_media_coord,
            '[SOMA]: media coordenadas': predadores_media_coord + prey_media_coord,

        }


        summary_rows.append(summary_row)

    df_summary = pd.DataFrame(summary_rows)

    # Substituir pontos por vírgulas nos valores numéricos
    for col in df_summary.select_dtypes(include=['float']).columns:
        df_summary[col] = df_summary[col].apply(lambda x: str(x).replace('.', ','))

    return df_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processar arquivos de simulação e gerar estatísticas.")
    parser.add_argument(
        "-i", "--input_dir", type=str, required=True,
        help="Diretório contendo os arquivos de simulação."
    )
    parser.add_argument(
        "-o", "--output_file", type=str, required=True,
        help="Caminho do arquivo CSV de saída."
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    file_paths = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith('sim')
    ]

    if not file_paths:
        print("Nenhum arquivo encontrado no diretório especificado.")
        exit(1)

    summary_df = process_files(file_paths)

    if not summary_df.empty:
        summary_df.to_csv(args.output_file, index=False, sep=';')
        print(f"Resumo salvo em: {args.output_file}")
    else:
        print("Nenhum dado processado. Verifique os arquivos de entrada.")
