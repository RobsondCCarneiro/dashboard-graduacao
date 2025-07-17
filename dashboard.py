import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import os
import base64
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Configuração da página Streamlit ---
st.set_page_config(layout="wide", page_title="Dashboard de Análise Acadêmica")

# --- Adicionar Imagem de Fundo (App e Sidebar) ---
background_image_app_path = "Background_app.jpg"
background_image_sidebar_path = "background_sidebar.jpg"

@st.cache_data
def get_base64_image(image_path):
    """Lê uma imagem e retorna sua representação em base64 para uso em CSS."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

encoded_background_app = get_base64_image(background_image_app_path)
encoded_background_sidebar = get_base64_image(background_image_sidebar_path)

css_string = """
<style>
/* Estilo para o fundo principal do aplicativo */
"""
if encoded_background_app:
    css_string += f"""
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_background_app}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    """
else:
    st.warning(f"A imagem de fundo do app '{background_image_app_path}' não foi encontrada. O fundo padrão será usado.")

css_string += """
/* Estilo para o fundo da sidebar */
"""
if encoded_background_sidebar:
    css_string += f"""
    .stSidebar {{
        background-image: url("data:image/jpeg;base64,{encoded_background_sidebar}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    """
else:
    st.warning(f"A imagem de fundo da sidebar '{background_image_sidebar_path}' não foi encontrada. O fundo padrão será usado.")

css_string += """
/* Ajustes de cor do texto para melhor legibilidade sobre fundos escuros */
/* Expandido para cobrir mais elementos de texto comuns */
.stMarkdown, .stText, .stLabel,
.stSelectbox label, .stMultiSelect label, .stSlider label, .stRadio label,
.stButton, .stProgress, .stExpander, 
.stAlert > div > div > div:nth-child(2) > div:first-child, /* Texto dentro de st.info, st.warning, st.error, st.success */
.stDataFrame .css-1qxtsqy.edgvbvh10, /* Cabeçalho de colunas do st.dataframe */
.stDataFrame .dataframe, /* Texto dentro do st.dataframe */
p, li { /* Parágrafos e itens de lista */
    color: white; /* Cor do texto padrão para branco */
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8); /* Sombra para melhorar contraste */
}

/* Cor de fonte vermelha para headers (h1, h2, h3, h4, h5, h6) com contorno preto */
h1, h2, h3, h4, h5, h6, .stHeader, .stSubheader, .stTitle {
    color: red !important; /* Cor de fonte vermelha */
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.9) !important; /* Contorno preto mais forte */
}


/* Ajustar a cor de fundo dos elementos internos do app principal para semitransparente */
.stApp > header, .stApp > div:first-child > div:nth-child(2) > div:nth-child(2) {
    background-color: rgba(0, 0, 0, 0.5); /* Fundo semi-transparente para o conteúdo principal */
    padding: 20px;
    border-radius: 10px;
}
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Ajustar a cor de fundo dos elementos internos da sidebar para semitransparente */
.stSidebar > div:first-child {
    background-color: rgba(0, 0, 0, 0.6); /* Fundo semi-transparente para o conteúdo da sidebar */
    padding: 10px;
    border-radius: 10px;
}
/* Ajustar os elementos de input dentro da sidebar */
.stSidebar .stSelectbox > div > div, .stSidebar .stMultiSelect > div > div {
    background-color: rgba(255, 255, 255, 0.1); /* Um pouco de transparência para elementos de input */
    border-radius: 5px;
}

/* Ajustes para as imagens de logo na sidebar */
.stSidebar img {
    background-color: transparent; /* Garante que o fundo das imagens seja transparente */
}

/* Estilo para os elementos de alerta (st.info, st.warning, st.error, st.success) */
/* Aumenta a opacidade do fundo para garantir que o texto branco com contorno preto se destaque */
.stAlert {
    background-color: rgba(0, 0, 0, 0.7) !important; /* Fundo mais opaco para alertas */
    color: white !important; /* Garante que o texto do alerta seja branco */
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.2); /* Borda sutil */
}
/* Cores do ícone do alerta */
.stAlert > div > div > div:nth-child(1) svg {
    fill: white !important; /* Ícone branco */
}

</style>
"""
st.markdown(css_string, unsafe_allow_html=True)


st.title("🎓 Dashboard de Análise Acadêmica 🎓")
st.markdown("Explore dados de ingressantes e egressos, filtrando por ano, nível de ensino, sexo, curso e unidade. **Novo!** Preveja o tempo de graduação com Machine Learning.")


# --- Caminhos para as pastas dos CSVs ---
INGRESSANTES_FOLDER = os.path.join("dataset", "ingressantes")
EGRESSOS_FOLDER = os.path.join("dataset", "egressos")

# Função auxiliar para extrair o ano do nome do arquivo
def extract_year_from_filename(filename):
    """Tenta extrair um ano (quatro dígitos) de uma string de nome de arquivo."""
    match = re.search(r'(\d{4})', filename)
    if match:
        return int(match.group(0))
    return None

@st.cache_data(show_spinner="Carregando e processando dados de INGRESSANTES...")
def load_and_preprocess_ingressantes_data():
    """
    Carrega todos os arquivos CSV de ingressantes, os concatena,
    adiciona a coluna 'ano' e padroniza as colunas 'nivel_ensino', 'sexo', 'nome_curso',
    'nome_unidade'.
    Retorna o DataFrame processado e uma lista de mensagens (sucesso/erro/aviso).
    """
    all_ingressantes_dfs = []
    messages = []

    if not os.path.exists(INGRESSANTES_FOLDER):
        messages.append({"type": "error", "text": f"Erro: A pasta de ingressantes '{INGRESSANTES_FOLDER}' não foi encontrada. "
                                                  "Certifique-se de que a estrutura é 'seu_app/dataset/ingressantes'."})
        return pd.DataFrame(), messages

    ingressantes_files = [f for f in os.listdir(INGRESSANTES_FOLDER) if f.endswith('.csv')]
    
    if not ingressantes_files:
        messages.append({"type": "warning", "text": f"Nenhum arquivo CSV encontrado na pasta '{INGRESSANTES_FOLDER}'."})
        return pd.DataFrame(), messages

    for file_name in ingressantes_files:
        file_path = os.path.join(INGRESSANTES_FOLDER, file_name)
        year = extract_year_from_filename(file_name)
        
        try:
            df = pd.read_csv(file_path, sep=';')
            df.columns = df.columns.str.lower() # Converte todos os nomes de colunas para minúsculas

            if year:
                df['ano'] = year
            else:
                messages.append({"type": "warning", "text": f"Não foi possível extrair o ano de '{file_name}'. O arquivo pode não ser incluído em filtros por ano."})

            all_ingressantes_dfs.append(df)
            messages.append({"type": "toast", "text": f"Carregado: {file_name}"})
        except Exception as e:
            messages.append({"type": "error", "text": f"Erro ao carregar o arquivo de ingressantes '{file_name}': {e}. Verifique o formato do CSV e a codificação."})
            continue

    if not all_ingressantes_dfs:
        messages.append({"type": "error", "text": "Nenhum dado de ingressantes pôde ser carregado. Retornando DataFrame vazio."})
        return pd.DataFrame(), messages

    df_ingressantes_combined = pd.concat(all_ingressantes_dfs, ignore_index=True)

    # --- Pré-processamento FINAIS para INGRESSANTES ---
    # Padroniza a coluna 'nivel_ensino'
    if 'nivel_ensino' in df_ingressantes_combined.columns:
        df_ingressantes_combined['nivel_ensino'] = df_ingressantes_combined['nivel_ensino'].astype(str).str.strip().str.upper()
        df_ingressantes_combined['nivel_ensino'].fillna('DESCONHECIDO', inplace=True)
    else:
        df_ingressantes_combined['nivel_ensino'] = 'DESCONHECIDO'
        messages.append({"type": "warning", "text": "Coluna 'nivel_ensino' não encontrada nos dados de ingressantes. Criando coluna 'nivel_ensino' com 'DESCONHECIDO'."})

    # Padroniza a coluna 'sexo'
    sexo_cols = [col for col in df_ingressantes_combined.columns if 'sexo' in col]
    if sexo_cols:
        df_ingressantes_combined['sexo'] = df_ingressantes_combined[sexo_cols[0]].astype(str).str.strip().str.upper()
        df_ingressantes_combined['sexo'] = df_ingressantes_combined['sexo'].replace({
            'MASCULINO': 'M', 'FEMININO': 'F', 'HOMEM': 'M', 'MULHER': 'F',
            'MALE': 'M', 'FEMALE': 'F'
        })
        df_ingressantes_combined['sexo'].fillna('INDEFINIDO', inplace=True)
        df_ingressantes_combined.loc[~df_ingressantes_combined['sexo'].isin(['M', 'F']), 'sexo'] = 'INDEFINIDO'
    else:
        df_ingressantes_combined['sexo'] = 'INDEFINIDO'
        messages.append({"type": "warning", "text": "Coluna de sexo não encontrada nos dados de ingressantes. Criando coluna 'sexo' com 'INDEFINIDO'."})

    # Padroniza a coluna 'nome_curso'
    if 'nome_curso' in df_ingressantes_combined.columns:
        df_ingressantes_combined['nome_curso'] = df_ingressantes_combined['nome_curso'].astype(str).str.strip().str.upper()
        df_ingressantes_combined['nome_curso'].fillna('DESCONHECIDO', inplace=True)
    else:
        df_ingressantes_combined['nome_curso'] = 'DESCONHECIDO'
        messages.append({"type": "warning", "text": "Coluna 'nome_curso' não encontrada nos dados de ingressantes. Criando coluna 'nome_curso' com 'DESCONHECIDO'."})

    # Padroniza a coluna 'nome_unidade' (Substitui 'nome_unidade_gestora')
    if 'nome_unidade' in df_ingressantes_combined.columns:
        df_ingressantes_combined['nome_unidade'] = df_ingressantes_combined['nome_unidade'].astype(str).str.strip().str.upper()
        df_ingressantes_combined['nome_unidade'].fillna('DESCONHECIDA', inplace=True)
    elif 'nome_unidade_gestora' in df_ingressantes_combined.columns: # Se existir a antiga, renomeia e padroniza
        df_ingressantes_combined['nome_unidade'] = df_ingressantes_combined['nome_unidade_gestora'].astype(str).str.strip().str.upper()
        df_ingressantes_combined['nome_unidade'].fillna('DESCONHECIDA', inplace=True)
        df_ingressantes_combined.drop(columns=['nome_unidade_gestora'], errors='ignore', inplace=True)
        messages.append({"type": "info", "text": "Coluna 'nome_unidade_gestora' renomeada para 'nome_unidade' nos dados de ingressantes."})
    else:
        df_ingressantes_combined['nome_unidade'] = 'DESCONHECIDA'
        messages.append({"type": "warning", "text": "Coluna 'nome_unidade' (ou 'nome_unidade_gestora') não encontrada nos dados de ingressantes. Criando coluna 'nome_unidade' com 'DESCONHECIDA'."})

    # 'total_periodos' não é aplicável para ingressantes, removendo qualquer referência
    if 'total_periodos' in df_ingressantes_combined.columns:
        df_ingressantes_combined.drop(columns=['total_periodos'], errors='ignore', inplace=True)


    df_ingressantes_combined.drop_duplicates(inplace=True)

    if 'ano' in df_ingressantes_combined.columns:
        df_ingressantes_combined['ano'] = pd.to_numeric(df_ingressantes_combined['ano'], errors='coerce').fillna(0).astype(int)
    else:
        messages.append({"type": "error", "text": "Coluna 'ano' não disponível nos dados de ingressantes para filtros. Verifique o nome dos arquivos."})
        
    messages.append({"type": "success", "text": "Dados de Ingressantes carregados e pré-processados!"})
    return df_ingressantes_combined, messages

@st.cache_data(show_spinner="Carregando e processando dados de EGRESSOS...")
def load_and_preprocess_egressos_data():
    """
    Carrega todos os arquivos CSV de egressos, os concatena,
    adiciona a coluna 'ano' e padroniza as colunas 'nivel_ensino', 'sexo', 'nome_curso',
    'nome_unidade' e CALCULA 'total_periodos'.
    Retorna o DataFrame processado e uma lista de mensagens.
    """
    all_egressos_dfs = []
    messages = []

    if not os.path.exists(EGRESSOS_FOLDER):
        messages.append({"type": "error", "text": f"Erro: A pasta de egressos '{EGRESSOS_FOLDER}' não foi encontrada. "
                                                 "Certifique-se de que a estrutura é 'seu_app/dataset/egressos'."})
        return pd.DataFrame(), messages

    egressos_files = [f for f in os.listdir(EGRESSOS_FOLDER) if f.endswith('.csv')]

    if not egressos_files:
        messages.append({"type": "warning", "text": f"Nenhum arquivo CSV encontrado na pasta '{EGRESSOS_FOLDER}'."})
        return pd.DataFrame(), messages

    for file_name in egressos_files:
        file_path = os.path.join(EGRESSOS_FOLDER, file_name)
        year = extract_year_from_filename(file_name)

        try:
            df = pd.read_csv(file_path, sep=';')
            df.columns = df.columns.str.lower()

            if year:
                df['ano'] = year
            else:
                messages.append({"type": "warning", "text": f"Não foi possível extrair o ano de '{file_name}'. O arquivo pode não ser incluído em filtros por ano."})
            
            all_egressos_dfs.append(df)
            messages.append({"type": "toast", "text": f"Carregado: {file_name}"})
        except Exception as e:
            messages.append({"type": "error", "text": f"Erro ao carregar o arquivo de egressos '{file_name}': {e}. Verifique o formato do CSV e a codificação."})
            continue

    if not all_egressos_dfs:
        messages.append({"type": "error", "text": "Nenhum dado de egressos pôde ser carregado. Retornando DataFrame vazio."})
        return pd.DataFrame(), messages

    df_egressos_combined = pd.concat(all_egressos_dfs, ignore_index=True)

    # --- Pré-processamento FINAIS para EGRESSOS ---
    # Padroniza a coluna 'nivel_ensino'
    if 'nivel_ensino' in df_egressos_combined.columns:
        df_egressos_combined['nivel_ensino'] = df_egressos_combined['nivel_ensino'].astype(str).str.strip().str.upper()
        df_egressos_combined['nivel_ensino'].fillna('DESCONHECIDO', inplace=True)
    else:
        df_egressos_combined['nivel_ensino'] = 'DESCONHECIDO'
        messages.append({"type": "warning", "text": "Coluna 'nivel_ensino' não encontrada nos dados de egressos. Criando coluna 'nivel_ensino' com 'DESCONHECIDO'."})

    # Padroniza a coluna 'sexo'
    sexo_cols = [col for col in df_egressos_combined.columns if 'sexo' in col]
    if sexo_cols:
        df_egressos_combined['sexo'] = df_egressos_combined[sexo_cols[0]].astype(str).str.strip().str.upper()
        df_egressos_combined['sexo'] = df_egressos_combined['sexo'].replace({
            'MASCULINO': 'M', 'FEMININO': 'F', 'HOMEM': 'M', 'MULHER': 'F',
            'MALE': 'M', 'FEMALE': 'F'
        })
        df_egressos_combined['sexo'].fillna('INDEFINIDO', inplace=True)
        df_egressos_combined.loc[~df_egressos_combined['sexo'].isin(['M', 'F']), 'sexo'] = 'INDEFINIDO'
    else:
        df_egressos_combined['sexo'] = 'INDEFINIDO'
        messages.append({"type": "warning", "text": "Coluna de sexo não encontrada nos dados de egressos. Criando coluna 'sexo' com 'INDEFINIDO'."})

    # Padroniza a coluna 'nome_curso'
    if 'nome_curso' in df_egressos_combined.columns:
        df_egressos_combined['nome_curso'] = df_egressos_combined['nome_curso'].astype(str).str.strip().str.upper()
        df_egressos_combined['nome_curso'].fillna('DESCONHECIDO', inplace=True)
    else:
        df_egressos_combined['nome_curso'] = 'DESCONHECIDO'
        messages.append({"type": "warning", "text": "Coluna 'nome_curso' não encontrada nos dados de egressos. Criando coluna 'nome_curso' com 'DESCONHECIDO'."})

    # Padroniza a coluna 'nome_unidade' (Substitui 'nome_unidade_gestora')
    if 'nome_unidade' in df_egressos_combined.columns:
        df_egressos_combined['nome_unidade'] = df_egressos_combined['nome_unidade'].astype(str).str.strip().str.upper()
        df_egressos_combined['nome_unidade'].fillna('DESCONHECIDA', inplace=True)
    elif 'nome_unidade_gestora' in df_egressos_combined.columns: # Se existir a antiga, renomeia e padroniza
        df_egressos_combined['nome_unidade'] = df_egressos_combined['nome_unidade_gestora'].astype(str).str.strip().str.upper()
        df_egressos_combined['nome_unidade'].fillna('DESCONHECIDA', inplace=True)
        df_egressos_combined.drop(columns=['nome_unidade_gestora'], errors='ignore', inplace=True)
        messages.append({"type": "info", "text": "Coluna 'nome_unidade_gestora' renomeada para 'nome_unidade' nos dados de egressos."})
    else:
        df_egressos_combined['nome_unidade'] = 'DESCONHECIDA'
        messages.append({"type": "warning", "text": "Coluna 'nome_unidade' (ou 'nome_unidade_gestora') não encontrada nos dados de egressos. Criando coluna 'nome_unidade' com 'DESCONHECIDA'."})

    # --- CALCULA total_periodos para EGRESSOS ---
    required_cols_for_periods = ['ano_conclusao', 'periodo_conclusao', 'ano_ingresso', 'periodo_ingresso']
    # Verifica se todas as colunas necessárias estão presentes no DataFrame antes de tentar o cálculo
    if all(col in df_egressos_combined.columns for col in required_cols_for_periods):
        try:
            # Converte para numérico e preenche NaN com 0 antes do cálculo
            for col in required_cols_for_periods:
                df_egressos_combined[col] = pd.to_numeric(df_egressos_combined[col], errors='coerce').fillna(0)
            
            # Garante que os períodos sejam inteiros para o cálculo
            df_egressos_combined['periodo_conclusao'] = df_egressos_combined['periodo_conclusao'].astype(int)
            df_egressos_combined['periodo_ingresso'] = df_egressos_combined['periodo_ingresso'].astype(int)
            df_egressos_combined['ano_conclusao'] = df_egressos_combined['ano_conclusao'].astype(int)
            df_egressos_combined['ano_ingresso'] = df_egressos_combined['ano_ingresso'].astype(int)

            # Cálculo do total de períodos
            df_egressos_combined['total_periodos'] = (
                (df_egressos_combined['ano_conclusao'] - df_egressos_combined['ano_ingresso']) * 2 +
                (df_egressos_combined['periodo_conclusao'] - df_egressos_combined['periodo_ingresso'])
            )
            # Garante que total_periodos não seja negativo (caso haja dados inconsistentes)
            df_egressos_combined['total_periodos'] = df_egressos_combined['total_periodos'].apply(lambda x: max(0, x))
            messages.append({"type": "success", "text": "Coluna 'total_periodos' calculada para egressos."})
        except Exception as e:
            df_egressos_combined['total_periodos'] = 0
            messages.append({"type": "warning", "text": f"Erro ao calcular 'total_periodos' para egressos: {e}. Coluna criada com 0."})
    else:
        df_egressos_combined['total_periodos'] = 0
        messages.append({"type": "warning", "text": f"Colunas ({', '.join(required_cols_for_periods)}) necessárias para calcular 'total_periodos' não encontradas nos dados de egressos. Coluna criada com 0."})


    df_egressos_combined.drop_duplicates(inplace=True)

    if 'ano' in df_egressos_combined.columns:
        df_egressos_combined['ano'] = pd.to_numeric(df_egressos_combined['ano'], errors='coerce').fillna(0).astype(int)
    else:
        messages.append({"type": "error", "text": "Coluna 'ano' não disponível nos dados de egressos para filtros. Verifique o nome dos arquivos."})

    messages.append({"type": "success", "text": "Dados de Egressos carregados e pré-processados!"})
    return df_egressos_combined, messages

# --- Carrega os DataFrames e as mensagens no início do seu app ---
df_ingressantes, ingressantes_load_messages = load_and_preprocess_ingressantes_data()
df_egressos, egressos_load_messages = load_and_preprocess_egressos_data()

# Exibe as mensagens coletadas APÓS a execução das funções cacheadas
for msg in ingressantes_load_messages:
    if msg["type"] == "error": st.error(msg["text"])
    elif msg["type"] == "warning": st.warning(msg["text"])
    elif msg["type"] == "success": st.success(msg["text"])
    elif msg["type"] == "toast": st.toast(msg["text"])

for msg in egressos_load_messages:
    if msg["type"] == "error": st.error(msg["text"])
    elif msg["type"] == "warning": st.warning(msg["text"])
    elif msg["type"] == "success": st.success(msg["text"])
    elif msg["type"] == "toast": st.toast(msg["text"])


# Colunas de logo na barra lateral
col1, col2 = st.sidebar.columns(2)
with col1:
    st.image("ufrn.png", width=150)
with col2:
    st.image("dca.png", width=100)

# --- Verificação inicial se os DataFrames foram carregados com sucesso ---
if df_ingressantes.empty and df_egressos.empty:
    st.error("Não foi possível carregar nenhum dado para o dashboard. "
             "Verifique as mensagens de erro acima, os caminhos das pastas, os nomes dos arquivos CSV, "
             "o delimitador (ponto e vírgula) e a existência/conteúdo das colunas essenciais.")
    st.stop()


# --- Filtros na Barra Lateral (Sidebar) ---
st.sidebar.header("Filtros Globais")

# Coleta todas as opções possíveis de nível de ensino dos dados brutos combinados
all_niveis_ensino_options = set()
if 'nivel_ensino' in df_ingressantes.columns:
    all_niveis_ensino_options.update(df_ingressantes['nivel_ensino'].unique())
if 'nivel_ensino' in df_egressos.columns:
    all_niveis_ensino_options.update(df_egressos['nivel_ensino'].unique())

sorted_niveis_ensino = sorted(list(all_niveis_ensino_options))
default_nivel_ensino_selection = sorted_niveis_ensino

selected_niveis_ensino = st.sidebar.multiselect(
    "Filtrar por Nível de Ensino:",
    options=sorted_niveis_ensino,
    default=default_nivel_ensino_selection,
    key='global_nivel_ensino_filter'
)

# Slider de Ano (2014-2024)
min_available_year = min(df_ingressantes['ano'].min() if not df_ingressantes.empty else 2014, 
                         df_egressos['ano'].min() if not df_egressos.empty else 2014)
max_available_year = max(df_ingressantes['ano'].max() if not df_ingressantes.empty else 2024, 
                         df_egressos['ano'].max() if not df_egressos.empty else 2024)

min_slider_year = max(2014, min_available_year)
max_slider_year = min(2024, max_available_year)

default_slider_value = (min_slider_year, max_slider_year)
if default_slider_value[0] > default_slider_value[1]:
    default_slider_value = (2014, 2024)

selected_years = st.sidebar.slider(
    'Intervalo de Anos:',
    min_value=min_slider_year,
    max_value=max_slider_year,
    value=default_slider_value,
    step=1,
    key='global_years_filter'
)

# Filtro por Sexo
all_sexos_options = set()
if 'sexo' in df_ingressantes.columns:
    all_sexos_options.update(df_ingressantes['sexo'].unique())
if 'sexo' in df_egressos.columns:
    all_sexos_options.update(df_egressos['sexo'].unique())

sorted_sexos = []
if 'M' in all_sexos_options: sorted_sexos.append('M')
if 'F' in all_sexos_options: sorted_sexos.append('F')
if 'INDEFINIDO' in all_sexos_options: sorted_sexos.append('INDEFINIDO')
for s in sorted(list(all_sexos_options)):
    if s not in ['M', 'F', 'INDEFINIDO']:
        sorted_sexos.append(s)

default_sex_selection = sorted_sexos if sorted_sexos else []

selected_sexos = st.sidebar.multiselect(
    "Filtrar por Sexo:",
    options=sorted_sexos,
    default=default_sex_selection,
    key='global_sex_filter'
)

# --- Lógica para Coletar Opções de Unidade (Aninhado: Nível + Ano) ---
# Cria DataFrames temporários combinados para obter as opções de unidade
# que respeitam os filtros de nível de ensino e ano já selecionados.
temp_df_for_unidade_options = pd.DataFrame()
if not df_ingressantes.empty:
    temp_df_for_unidade_options = pd.concat([temp_df_for_unidade_options, df_ingressantes], ignore_index=True)
if not df_egressos.empty:
    temp_df_for_unidade_options = pd.concat([temp_df_for_unidade_options, df_egressos], ignore_index=True)

if not temp_df_for_unidade_options.empty:
    # Aplica filtro de nível de ensino
    if selected_niveis_ensino and 'nivel_ensino' in temp_df_for_unidade_options.columns:
        temp_df_for_unidade_options = temp_df_for_unidade_options[
            temp_df_for_unidade_options['nivel_ensino'].isin(selected_niveis_ensino)
        ]
    else: 
        temp_df_for_unidade_options = pd.DataFrame() 

    # Aplica filtro de ano
    if selected_years and 'ano' in temp_df_for_unidade_options.columns and not temp_df_for_unidade_options.empty:
        min_y_sel, max_y_sel = selected_years
        temp_df_for_unidade_options = temp_df_for_unidade_options[
            (temp_df_for_unidade_options['ano'] >= min_y_sel) & 
            (temp_df_for_unidade_options['ano'] <= max_y_sel)
        ]
    elif not temp_df_for_unidade_options.empty and 'ano' not in temp_df_for_unidade_options.columns:
        st.warning("Coluna 'ano' não encontrada nos dados para filtro de unidade. Opções de unidade podem ser imprecisas.")
    
all_unidades_options = set()
if 'nome_unidade' in temp_df_for_unidade_options.columns and not temp_df_for_unidade_options.empty:
    all_unidades_options.update(temp_df_for_unidade_options['nome_unidade'].unique())

sorted_unidades = sorted(list(all_unidades_options))
default_unidade_selection = sorted_unidades 

selected_unidades = st.sidebar.multiselect(
    "Filtrar por Unidade:",
    options=sorted_unidades,
    default=default_unidade_selection,
    key='global_unidade_filter'
)

# --- Lógica para Coletar Opções de Curso (Aninhado: Nível + Ano + Unidade) ---
# O DataFrame base para as opções de curso agora é o 'temp_df_for_unidade_options'
# que já foi filtrado por Nível e Ano. Agora, filtramos por Unidade.
temp_df_for_course_options = temp_df_for_unidade_options.copy() 

# Aplica filtro de unidade ao DataFrame temporário
if selected_unidades and 'nome_unidade' in temp_df_for_course_options.columns and not temp_df_for_course_options.empty:
    temp_df_for_course_options = temp_df_for_course_options[
        temp_df_for_course_options['nome_unidade'].isin(selected_unidades)
    ]
else: # Se nenhuma unidade selecionada ou coluna inexistente, não há cursos para exibir
    temp_df_for_course_options = pd.DataFrame()


all_cursos_options = set()
if 'nome_curso' in temp_df_for_course_options.columns and not temp_df_for_course_options.empty:
    all_cursos_options.update(temp_df_for_course_options['nome_curso'].unique())

sorted_cursos = sorted(list(all_cursos_options))
default_curso_selection = sorted_cursos

selected_cursos = st.sidebar.multiselect(
    "Filtrar por Curso:",
    options=sorted_cursos,
    default=default_curso_selection,
    key='global_course_filter'
)


# --- Função para aplicar os filtros da sidebar aos DataFrames ---
def apply_sidebar_filters(df, years_range, sex_filter_list, course_filter_list, nivel_ensino_filter_list, unidade_filter_list):
    """Aplica os filtros de ano, sexo, curso, nível de ensino e unidade a um DataFrame dado."""
    df_filtered = df.copy()

    # Filtro por Nível de Ensino - PRIMEIRO
    if 'nivel_ensino' in df_filtered.columns and nivel_ensino_filter_list:
        df_filtered = df_filtered[df_filtered['nivel_ensino'].isin(nivel_ensino_filter_list)]
    else:
        return pd.DataFrame() 
    
    # Filtro por Ano - SEGUNDO
    min_year_sel, max_year_sel = years_range
    if 'ano' in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered['ano'] >= min_year_sel) & 
            (df_filtered['ano'] <= max_y_sel)
        ]
    else:
        st.warning(f"Coluna 'ano' não encontrada no DataFrame de {df.name if hasattr(df, 'name') else 'dados'} para filtro de ano.")

    # Filtro por Unidade - TERCEIRO (Ordem alterada)
    if 'nome_unidade' in df_filtered.columns and unidade_filter_list:
        df_filtered = df_filtered[df_filtered['nome_unidade'].isin(unidade_filter_list)]
    elif not unidade_filter_list and 'nome_unidade' in df_filtered.columns:
        pass 

    # Filtro por Nome do Curso - QUARTO (Ordem alterada)
    if 'nome_curso' in df_filtered.columns and course_filter_list:
        df_filtered = df_filtered[df_filtered['nome_curso'].isin(course_filter_list)]
    elif not course_filter_list and 'nome_curso' in df_filtered.columns:
        pass 

    # Filtro por Sexo - QUINTO
    if 'sexo' in df_filtered.columns and sex_filter_list:
        df_filtered = df_filtered[df_filtered['sexo'].isin(sex_filter_list)]
    elif not sex_filter_list and 'sexo' in df_filtered.columns:
        pass 


    return df_filtered

# Aplica os filtros da sidebar aos DataFrames de ingressantes e egressos
df_ingressantes.name = "ingressantes" 
df_egressos.name = "egressos"

filtered_ingressantes = apply_sidebar_filters(
    df_ingressantes, selected_years, selected_sexos, 
    selected_cursos, selected_niveis_ensino, selected_unidades
)
filtered_egressos = apply_sidebar_filters(
    df_egressos, selected_years, selected_sexos, 
    selected_cursos, selected_niveis_ensino, selected_unidades
)

# Apply sexo_rotulo mapping to filtered_egressos for consistent use in plots
if 'sexo' in filtered_egressos.columns:
    sexo_map = {'M': 'Masculino', 'F': 'Feminino', 'INDEFINIDO': 'Não Informado'}
    filtered_egressos['sexo_rotulo'] = filtered_egressos['sexo'].map(sexo_map).fillna('Não Informado')
else:
    filtered_egressos['sexo_rotulo'] = 'Não Informado' # Fallback for missing sexo column
    st.info("Coluna 'sexo' não disponível nos dados filtrados para rótulos de sexo. Usando 'Não Informado'.")


# --- Geração e Exibição dos Gráficos com Plotly.express em ABAS ---
tab_ingressantes_viz, tab_egressos_viz, tab_comparacao_viz, tab_ml_classificacao, tab_ml_regressao = st.tabs([
    "Análise de Ingressantes",
    "Análise de Egressos",
    "Comparativo Geral",
    "Predição de Desempenho (Classificação)", # Nova aba para o modelo de classificação
    "Predição de Duração de Curso (Regressão)" # Nova aba para o modelo de regressão
])

# --- TAB 1: Análise de Ingressantes ---
with tab_ingressantes_viz:
    st.header("Análise Detalhada de Alunos Ingressantes")

    # --- NOVO GRÁFICO 1: Rosca Aninhada - Unidade > Nível de Ensino > Curso ---
    st.subheader("Distribuição Hierárquica de Ingressantes: Unidade > Nível de Ensino > Curso")

    required_cols_unidade_nivel_curso = ['nome_unidade', 'nivel_ensino', 'nome_curso']
    if all(col in filtered_ingressantes.columns for col in required_cols_unidade_nivel_curso) and not filtered_ingressantes.empty:
        # Criar dados para o Sunburst Chart
        df_unidade_nivel_curso = filtered_ingressantes.groupby(required_cols_unidade_nivel_curso).size().reset_index(name='count')

        # Para o Sunburst, precisamos de IDs únicos e pais
        # Cada nó terá um ID (Unidade, Unidade_Nivel, Unidade_Nivel_Curso)
        # O pai de Unidade_Nivel é Unidade, o pai de Unidade_Nivel_Curso é Unidade_Nivel

        # Nível 1: Unidade
        unidades = df_unidade_nivel_curso.groupby('nome_unidade')['count'].sum().reset_index()
        unidades['ids'] = unidades['nome_unidade']
        unidades['parents'] = ""
        unidades['labels'] = unidades['nome_unidade']

        # Nível 2: Nível de Ensino dentro de Unidade
        unidade_nivel = df_unidade_nivel_curso.groupby(['nome_unidade', 'nivel_ensino'])['count'].sum().reset_index()
        unidade_nivel['ids'] = unidade_nivel['nome_unidade'] + " - " + unidade_nivel['nivel_ensino']
        unidade_nivel['parents'] = unidade_nivel['nome_unidade']
        unidade_nivel['labels'] = unidade_nivel['nivel_ensino']

        # Nível 3: Curso dentro de Nível de Ensino (dentro de Unidade)
        cursos_completos = df_unidade_nivel_curso.copy()
        cursos_completos['ids'] = cursos_completos['nome_unidade'] + " - " + cursos_completos['nivel_ensino'] + " - " + cursos_completos['nome_curso']
        cursos_completos['parents'] = cursos_completos['nome_unidade'] + " - " + cursos_completos['nivel_ensino']
        cursos_completos['labels'] = cursos_completos['nome_curso']

        # Concatena todos os níveis para o Sunburst
        sunburst_data_unidade = pd.concat([
            unidades[['ids', 'parents', 'labels', 'count']],
            unidade_nivel[['ids', 'parents', 'labels', 'count']],
            cursos_completos[['ids', 'parents', 'labels', 'count']]
        ])

        fig_sunburst_unidade = go.Figure(go.Sunburst(
            ids=sunburst_data_unidade['ids'],
            labels=sunburst_data_unidade['labels'],
            parents=sunburst_data_unidade['parents'],
            values=sunburst_data_unidade['count'],
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Alunos: %{value}<br>Percentual: %{percentParent}<extra></extra>'
        ))
        fig_sunburst_unidade.update_layout(margin = dict(t=0, l=0, r=0, b=0),
                                            title_text='Ingressantes por Unidade, Nível de Ensino e Curso')
        st.plotly_chart(fig_sunburst_unidade, use_container_width=True)
    else:
        st.info("Colunas 'nome_unidade', 'nivel_ensino' ou 'nome_curso' não disponíveis nos dados filtrados de ingressantes para este gráfico, ou DataFrame vazio.")

    st.markdown("---") # Separador para o próximo gráfico

    # --- NOVO GRÁFICO 2: Rosca Aninhada - Sexo > Nível de Ensino > Curso ---
    st.subheader("Distribuição Hierárquica de Ingressantes: Sexo > Nível de Ensino > Curso")

    required_cols_sexo_nivel_curso = ['sexo', 'nivel_ensino', 'nome_curso']
    if all(col in filtered_ingressantes.columns for col in required_cols_sexo_nivel_curso) and not filtered_ingressantes.empty:
        # Criar dados para o Sunburst Chart
        df_sexo_nivel_curso = filtered_ingressantes.groupby(required_cols_sexo_nivel_curso).size().reset_index(name='count')

        # Nível 1: Sexo
        sexos = df_sexo_nivel_curso.groupby('sexo')['count'].sum().reset_index()
        sexos['ids'] = sexos['sexo']
        sexos['parents'] = ""
        sexos['labels'] = sexos['sexo']

        # Nível 2: Nível de Ensino dentro de Sexo
        sexo_nivel = df_sexo_nivel_curso.groupby(['sexo', 'nivel_ensino'])['count'].sum().reset_index()
        sexo_nivel['ids'] = sexo_nivel['sexo'] + " - " + sexo_nivel['nivel_ensino']
        sexo_nivel['parents'] = sexo_nivel['sexo']
        sexo_nivel['labels'] = sexo_nivel['nivel_ensino']

        # Nível 3: Curso dentro de Nível de Ensino (dentro de Sexo)
        cursos_completos_sexo = df_sexo_nivel_curso.copy()
        cursos_completos_sexo['ids'] = cursos_completos_sexo['sexo'] + " - " + cursos_completos_sexo['nivel_ensino'] + " - " + cursos_completos_sexo['nome_curso']
        cursos_completos_sexo['parents'] = cursos_completos_sexo['sexo'] + " - " + cursos_completos_sexo['nivel_ensino']
        cursos_completos_sexo['labels'] = cursos_completos_sexo['nome_curso']

        # Concatena todos os níveis para o Sunburst
        sunburst_data_sexo = pd.concat([
            sexos[['ids', 'parents', 'labels', 'count']],
            sexo_nivel[['ids', 'parents', 'labels', 'count']],
            cursos_completos_sexo[['ids', 'parents', 'labels', 'count']]
        ])

        fig_sunburst_sexo = go.Figure(go.Sunburst(
            ids=sunburst_data_sexo['ids'],
            labels=sunburst_data_sexo['labels'],
            parents=sunburst_data_sexo['parents'],
            values=sunburst_data_sexo['count'],
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Alunos: %{value}<br>Percentual: %{percentParent}<extra></extra>'
        ))
        fig_sunburst_sexo.update_layout(margin = dict(t=0, l=0, r=0, b=0),
                                         title_text='Ingressantes por Sexo, Nível de Ensino e Curso')
        st.plotly_chart(fig_sunburst_sexo, use_container_width=True)
    else:
        st.info("Colunas 'sexo', 'nivel_ensino' ou 'nome_curso' não disponíveis nos dados filtrados de ingressantes para este gráfico, ou DataFrame vazio.")


    st.markdown("---") # Separador para o próximo gráfico

    # --- GRÁFICO ANTERIOR: Rosca Aninhada - Nível de Ensino > Curso (primeira sugestão) ---
    st.subheader("Distribuição Hierárquica: Nível de Ensino e Cursos (Ingressantes)")

    if 'nivel_ensino' in filtered_ingressantes.columns and 'nome_curso' in filtered_ingressantes.columns and not filtered_ingressantes.empty:
        df_grouped_hierarchy = filtered_ingressantes.groupby(['nivel_ensino', 'nome_curso']).size().reset_index(name='count')
        
        # Para o Sunburst Chart (Hierarquia de Nível de Ensino -> Curso)
        # Nível 1: Nível de Ensino
        niveis = df_grouped_hierarchy.groupby('nivel_ensino')['count'].sum().reset_index()
        niveis['ids'] = niveis['nivel_ensino']
        niveis['parents'] = ""
        niveis['labels'] = niveis['nivel_ensino']

        # Nível 2: Curso dentro de Nível de Ensino
        cursos_no_nivel = df_grouped_hierarchy.copy()
        cursos_no_nivel['ids'] = cursos_no_nivel['nivel_ensino'] + " - " + cursos_no_nivel['nome_curso']
        cursos_no_nivel['parents'] = cursos_no_nivel['nivel_ensino']
        cursos_no_nivel['labels'] = cursos_no_nivel['nome_curso']

        sunburst_data_nivel = pd.concat([
            niveis[['ids', 'parents', 'labels', 'count']],
            cursos_no_nivel[['ids', 'parents', 'labels', 'count']]
        ])

        fig_donut_aninhado = go.Figure(go.Sunburst(
            ids=sunburst_data_nivel['ids'],
            labels=sunburst_data_nivel['labels'],
            parents=sunburst_data_nivel['parents'],
            values=sunburst_data_nivel['count'],
            branchvalues="total", 
            hovertemplate='<b>%{label}</b><br>Alunos: %{value}<br>Percentual: %{percentParent}<extra></extra>'
        ))
        fig_donut_aninhado.update_layout(
            margin = dict(t=0, l=0, r=0, b=0),
            title_text='Ingressantes por Nível de Ensino e Cursos'
        )

        st.plotly_chart(fig_donut_aninhado, use_container_width=True)

    else:
        st.info("Colunas 'nivel_ensino' ou 'nome_curso' não disponíveis nos dados filtrados de ingressantes para o gráfico de rosca aninhado, ou DataFrame vazio.")


    st.markdown("---") # Separador para os próximos gráficos


    if not filtered_ingressantes.empty:
        with st.container():
            col_ing1, col_ing2 = st.columns(2)

            with col_ing1:
                st.subheader("Ingressantes por Ano")
                if 'ano' in filtered_ingressantes.columns:
                    ingressantes_por_ano = filtered_ingressantes.groupby('ano').size().reset_index(name='count')
                    fig_ing_ano = px.bar(ingressantes_por_ano, x='ano', y='count',
                                         title='Número de Ingressantes por Ano',
                                         labels={'ano': 'Ano de Ingresso', 'count': 'Número de Alunos'})
                    fig_ing_ano.update_xaxes(dtick=1, tickformat="%Y")
                    st.plotly_chart(fig_ing_ano, use_container_width=True)
                else:
                    st.info("Coluna 'ano' não disponível nos dados de ingressantes para este gráfico.")

            with col_ing2:
                st.subheader("Distribuição de Sexo")
                if 'sexo' in filtered_ingressantes.columns:
                    sexo_dist_ing = filtered_ingressantes['sexo'].value_counts(normalize=True).reset_index()
                    sexo_dist_ing.columns = ['sexo', 'percentage']
                    sexo_dist_ing['percentage'] = sexo_dist_ing['percentage'] * 100
                    fig_ing_sexo = px.pie(sexo_dist_ing, names='sexo', values='percentage',
                                          title='Distribuição Percentual de Sexo',
                                          hole=0.3)
                    st.plotly_chart(fig_ing_sexo, use_container_width=True)
                else:
                    st.info("Coluna 'sexo' não disponível nos dados de ingressantes para este gráfico.")
        
        st.subheader("Tabela de Dados Filtrados (Ingressantes)")
        
        # Adiciona o slider para controlar a quantidade de alunos
        max_rows_ingressantes = len(filtered_ingressantes)
        num_alunos_ingressantes = st.slider(
            "Número de alunos a exibir (Ingressantes):",
            min_value=1,
            max_value=max_rows_ingressantes if max_rows_ingressantes > 0 else 1,
            value=min(10, max_rows_ingressantes) if max_rows_ingressantes > 0 else 1,
            step=1,
            key='num_alunos_ingressantes_slider'
        )

        # Cria uma cópia para remover a coluna sem afetar o DataFrame original
        df_display_ingressantes = filtered_ingressantes.copy()
        if 'nome_discente' in df_display_ingressantes.columns:
            df_display_ingressantes = df_display_ingressantes.drop(columns=['nome_discente'])

        st.dataframe(df_display_ingressantes.head(num_alunos_ingressantes))
        st.write(f"Total de registros de Ingressantes filtrados: {len(filtered_ingressantes)}")

    else:
        st.info("Nenhum dado de ingressantes disponível com os filtros selecionados para análise.")

# --- TAB 2: Análise de Egressos ---
with tab_egressos_viz:
    st.header("Análise Detalhada de Alunos Egressos")

    # --- NOVO GRÁFICO 1: Rosca Aninhada - Unidade > Nível de Ensino > Curso (EGRESSOS) ---
    st.subheader("Distribuição Hierárquica de Egressos: Unidade > Nível de Ensino > Curso")

    required_cols_unidade_nivel_curso_egressos = ['nome_unidade', 'nivel_ensino', 'nome_curso']
    if all(col in filtered_egressos.columns for col in required_cols_unidade_nivel_curso_egressos) and not filtered_egressos.empty:
        df_unidade_nivel_curso_egressos = filtered_egressos.groupby(required_cols_unidade_nivel_curso_egressos).size().reset_index(name='count')

        # Nível 1: Unidade
        unidades_egressos = df_unidade_nivel_curso_egressos.groupby('nome_unidade')['count'].sum().reset_index()
        unidades_egressos['ids'] = unidades_egressos['nome_unidade']
        unidades_egressos['parents'] = ""
        unidades_egressos['labels'] = unidades_egressos['nome_unidade']

        # Nível 2: Nível de Ensino dentro de Unidade
        unidade_nivel_egressos = df_unidade_nivel_curso_egressos.groupby(['nome_unidade', 'nivel_ensino'])['count'].sum().reset_index()
        unidade_nivel_egressos['ids'] = unidade_nivel_egressos['nome_unidade'] + " - " + unidade_nivel_egressos['nivel_ensino']
        unidade_nivel_egressos['parents'] = unidade_nivel_egressos['nome_unidade']
        unidade_nivel_egressos['labels'] = unidade_nivel_egressos['nivel_ensino']

        # Nível 3: Curso dentro de Nível de Ensino (dentro de Unidade)
        cursos_completos_egressos = df_unidade_nivel_curso_egressos.copy()
        cursos_completos_egressos['ids'] = cursos_completos_egressos['nome_unidade'] + " - " + cursos_completos_egressos['nivel_ensino'] + " - " + cursos_completos_egressos['nome_curso']
        cursos_completos_egressos['parents'] = cursos_completos_egressos['nome_unidade'] + " - " + cursos_completos_egressos['nivel_ensino']
        cursos_completos_egressos['labels'] = cursos_completos_egressos['nome_curso']

        # Concatena todos os níveis para o Sunburst
        sunburst_data_unidade_egressos = pd.concat([
            unidades_egressos[['ids', 'parents', 'labels', 'count']],
            unidade_nivel_egressos[['ids', 'parents', 'labels', 'count']],
            cursos_completos_egressos[['ids', 'parents', 'labels', 'count']]
        ])

        fig_sunburst_unidade_egressos = go.Figure(go.Sunburst(
            ids=sunburst_data_unidade_egressos['ids'],
            labels=sunburst_data_unidade_egressos['labels'],
            parents=sunburst_data_unidade_egressos['parents'],
            values=sunburst_data_unidade_egressos['count'],
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Alunos: %{value}<br>Percentual: %{percentParent}<extra></extra>'
        ))
        fig_sunburst_unidade_egressos.update_layout(margin = dict(t=0, l=0, r=0, b=0),
                                                    title_text='Egressos por Unidade, Nível de Ensino e Curso')
        st.plotly_chart(fig_sunburst_unidade_egressos, use_container_width=True)
    else:
        st.info("Colunas 'nome_unidade', 'nivel_ensino' ou 'nome_curso' não disponíveis nos dados filtrados de egressos para este gráfico, ou DataFrame vazio.")

    st.markdown("---") # Separador

    # --- NOVO GRÁFICO 2: Rosca Aninhada - Sexo > Nível de Ensino > Curso (EGRESSOS) ---
    st.subheader("Distribuição Hierárquica de Egressos: Sexo > Nível de Ensino > Curso")

    required_cols_sexo_nivel_curso_egressos = ['sexo', 'nivel_ensino', 'nome_curso']
    if all(col in filtered_egressos.columns for col in required_cols_sexo_nivel_curso_egressos) and not filtered_egressos.empty:
        df_sexo_nivel_curso_egressos = filtered_egressos.groupby(required_cols_sexo_nivel_curso_egressos).size().reset_index(name='count')

        # Nível 1: Sexo
        sexos_egressos = df_sexo_nivel_curso_egressos.groupby('sexo')['count'].sum().reset_index()
        sexos_egressos['ids'] = sexos_egressos['sexo']
        sexos_egressos['parents'] = ""
        sexos_egressos['labels'] = sexos_egressos['sexo']

        # Nível 2: Nível de Ensino dentro de Sexo
        sexo_nivel_egressos = df_sexo_nivel_curso_egressos.groupby(['sexo', 'nivel_ensino'])['count'].sum().reset_index()
        sexo_nivel_egressos['ids'] = sexo_nivel_egressos['sexo'] + " - " + sexo_nivel_egressos['nivel_ensino']
        sexo_nivel_egressos['parents'] = sexo_nivel_egressos['sexo']
        sexo_nivel_egressos['labels'] = sexo_nivel_egressos['nivel_ensino']

        # Nível 3: Curso dentro de Nível de Ensino (dentro de Sexo)
        cursos_completos_sexo_egressos = df_sexo_nivel_curso_egressos.copy()
        cursos_completos_sexo_egressos['ids'] = cursos_completos_sexo_egressos['sexo'] + " - " + cursos_completos_sexo_egressos['nivel_ensino'] + " - " + cursos_completos_sexo_egressos['nome_curso']
        cursos_completos_sexo_egressos['parents'] = cursos_completos_sexo_egressos['sexo'] + " - " + cursos_completos_sexo_egressos['nivel_ensino']
        cursos_completos_sexo_egressos['labels'] = cursos_completos_sexo_egressos['nome_curso']

        # Concatena todos os níveis para o Sunburst
        sunburst_data_sexo_egressos = pd.concat([
            sexos_egressos[['ids', 'parents', 'labels', 'count']],
            sexo_nivel_egressos[['ids', 'parents', 'labels', 'count']],
            cursos_completos_sexo_egressos[['ids', 'parents', 'labels', 'count']]
        ])

        fig_sunburst_sexo_egressos = go.Figure(go.Sunburst(
            ids=sunburst_data_sexo_egressos['ids'],
            labels=sunburst_data_sexo_egressos['labels'],
            parents=sunburst_data_sexo_egressos['parents'],
            values=sunburst_data_sexo_egressos['count'],
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Alunos: %{value}<br>Percentual: %{percentParent}<extra></extra>'
        ))
        fig_sunburst_sexo_egressos.update_layout(margin = dict(t=0, l=0, r=0, b=0),
                                                 title_text='Egressos por Sexo, Nível de Ensino e Curso')
        st.plotly_chart(fig_sunburst_sexo_egressos, use_container_width=True)
    else:
        st.info("Colunas 'sexo', 'nivel_ensino' ou 'nome_curso' não disponíveis nos dados filtrados de egressos para este gráfico, ou DataFrame vazio.")

    st.markdown("---") # Separador

    # --- GRÁFICO: Rosca Aninhada - Nível de Ensino > Curso (EGRESSOS) ---
    st.subheader("Distribuição Hierárquica: Nível de Ensino e Cursos (Egressos)")

    if 'nivel_ensino' in filtered_egressos.columns and 'nome_curso' in filtered_egressos.columns and not filtered_egressos.empty:
        df_grouped_hierarchy_egressos = filtered_egressos.groupby(['nivel_ensino', 'nome_curso']).size().reset_index(name='count')
        
        # Para o Sunburst Chart (Hierarquia de Nível de Ensino -> Curso)
        # Nível 1: Nível de Ensino
        niveis_egressos = df_grouped_hierarchy_egressos.groupby('nivel_ensino')['count'].sum().reset_index()
        niveis_egressos['ids'] = niveis_egressos['nivel_ensino']
        niveis_egressos['parents'] = ""
        niveis_egressos['labels'] = niveis_egressos['nivel_ensino']

        # Nível 2: Curso dentro de Nível de Ensino
        cursos_no_nivel_egressos = df_grouped_hierarchy_egressos.copy()
        cursos_no_nivel_egressos['ids'] = cursos_no_nivel_egressos['nivel_ensino'] + " - " + cursos_no_nivel_egressos['nome_curso']
        cursos_no_nivel_egressos['parents'] = cursos_no_nivel_egressos['nivel_ensino']
        cursos_no_nivel_egressos['labels'] = cursos_no_nivel_egressos['nome_curso']

        sunburst_data_nivel_egressos = pd.concat([
            niveis_egressos[['ids', 'parents', 'labels', 'count']],
            cursos_no_nivel_egressos[['ids', 'parents', 'labels', 'count']]
        ])

        fig_donut_aninhado_egressos = go.Figure(go.Sunburst(
            ids=sunburst_data_nivel_egressos['ids'],
            labels=sunburst_data_nivel_egressos['labels'],
            parents=sunburst_data_nivel_egressos['parents'],
            values=sunburst_data_nivel_egressos['count'],
            branchvalues="total", 
            hovertemplate='<b>%{label}</b><br>Alunos: %{value}<br>Percentual: %{percentParent}<extra></extra>'
        ))
        fig_donut_aninhado_egressos.update_layout(
            margin = dict(t=0, l=0, r=0, b=0),
            title_text='Egressos por Nível de Ensino e Cursos'
        )

        st.plotly_chart(fig_donut_aninhado_egressos, use_container_width=True)

    else:
        st.info("Colunas 'nivel_ensino' ou 'nome_curso' não disponíveis nos dados filtrados de egressos para o gráfico de rosca aninhado, ou DataFrame vazio.")


    st.markdown("---") # Separador para os próximos gráficos


    if not filtered_egressos.empty:
        with st.container():
            col_eg1, col_eg2 = st.columns(2)

            with col_eg1:
                st.subheader("Egressos por Ano de Conclusão")
                if 'ano' in filtered_egressos.columns:
                    egressos_por_ano = filtered_egressos.groupby('ano').size().reset_index(name='count')
                    fig_eg_ano = px.bar(egressos_por_ano, x='ano', y='count',
                                        title='Número de Egressos por Ano de Conclusão',
                                        labels={'ano': 'Ano de Conclusão', 'count': 'Número de Alunos'})
                    fig_eg_ano.update_xaxes(dtick=1, tickformat="%Y")
                    st.plotly_chart(fig_eg_ano, use_container_width=True)
                else:
                    st.info("Coluna 'ano' não disponível nos dados de egressos para este gráfico.")

            with col_eg2:
                st.subheader("Distribuição de Sexo")
                if 'sexo' in filtered_egressos.columns:
                    sexo_dist_eg = filtered_egressos['sexo'].value_counts(normalize=True).reset_index()
                    sexo_dist_eg.columns = ['sexo', 'percentage']
                    sexo_dist_eg['percentage'] = sexo_dist_eg['percentage'] * 100
                    fig_eg_sexo = px.pie(sexo_dist_eg, names='sexo', values='percentage',
                                         title='Distribuição Percentual de Sexo',
                                         hole=0.3)
                    st.plotly_chart(fig_eg_sexo, use_container_width=True)
                else:
                    st.info("Coluna 'sexo' não disponível nos dados de egressos para este gráfico.")
        
        st.subheader("Tabela de Dados Filtrados (Egressos)")

        # Adiciona o slider para controlar a quantidade de alunos
        max_rows_egressos = len(filtered_egressos)
        num_alunos_egressos = st.slider(
            "Número de alunos a exibir (Egressos):",
            min_value=1,
            max_value=max_rows_egressos if max_rows_egressos > 0 else 1,
            value=min(10, max_rows_egressos) if max_rows_egressos > 0 else 1,
            step=1,
            key='num_alunos_egressos_slider'
        )

        # Cria uma cópia para remover a coluna sem afetar o DataFrame original
        df_display_egressos = filtered_egressos.copy()
        if 'nome_discente' in df_display_egressos.columns:
            df_display_egressos = df_display_egressos.drop(columns=['nome_discente'])

        st.dataframe(df_display_egressos.head(num_alunos_egressos))
        st.write(f"Total de registros de Egressos filtrados: {len(filtered_egressos)}")

        st.markdown("---") # Separador para o próximo gráfico

        # --- GRÁFICO: Violino de Semestres Concluídos ---
        st.subheader("Distribuição do Total de Semestres Concluídos por Unidade e Sexo")
        
        LIMITE_MAX_PERIODOS = st.slider(
            "Limite Máximo de Semestres para Gráfico de Violino:",
            min_value=1,
            max_value=30, # Ajuste este max conforme a distribuição real dos seus dados
            value=20,
            step=1,
            key='violin_periods_limit_egressos' # Chave única para este slider na aba de egressos
        )

        df_egressos_plot_for_violin = filtered_egressos.copy()
        
        # --- Verificações de colunas e aplicação de filtros específicos para o gráfico de violino ---
        missing_violin_cols = []
        if 'total_periodos' not in df_egressos_plot_for_violin.columns:
            missing_violin_cols.append('total_periodos')
        if 'nome_unidade' not in df_egressos_plot_for_violin.columns: 
            missing_violin_cols.append('nome_unidade') 
        if 'sexo_rotulo' not in df_egressos_plot_for_violin.columns:
            missing_violin_cols.append('sexo_rotulo')

        if missing_violin_cols:
            st.info(f"As seguintes colunas essenciais para o gráfico de violino não foram encontradas nos dados filtrados: {', '.join(missing_violin_cols)}. O gráfico não será exibido. Verifique se os dados de egressos contêm essas colunas após o carregamento e filtros.")
            df_egressos_plot_for_violin = pd.DataFrame() 
        
        # Aplica o filtro de limite de períodos e remove 0 períodos (se o DataFrame não estiver vazio)
        if not df_egressos_plot_for_violin.empty:
            df_egressos_plot_for_violin = df_egressos_plot_for_violin[
                (df_egressos_plot_for_violin['total_periodos'] <= LIMITE_MAX_PERIODOS) &
                (df_egressos_plot_for_violin['total_periodos'] > 0) 
            ]
        
        # --- Verificação final e plotagem ---
        if not df_egressos_plot_for_violin.empty and \
           'nome_unidade' in df_egressos_plot_for_violin.columns and \
           'total_periodos' in df_egressos_plot_for_violin.columns and \
           'sexo_rotulo' in df_egressos_plot_for_violin.columns:
            
            fig_violin = px.violin(
                df_egressos_plot_for_violin,
                x='nome_unidade',
                y='total_periodos',
                color='sexo_rotulo',
                box=True,
                points="outliers",
                title=f'Distribuição do Total de Semestres Concluídos por Unidade e Sexo (Máx {LIMITE_MAX_PERIODOS} Semestres)',
                labels={'nome_unidade': 'Unidade', 'total_periodos': 'Total de Semestres Concluídos', 'sexo_rotulo': 'Gênero'},
                hover_data={'total_periodos': True, 'nome_curso': True, 'ano': True},
                height=600,
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_violin.update_xaxes(tickangle=45) 
            st.plotly_chart(fig_violin, use_container_width=True)
        else:
            st.info("Nenhum dado disponível para o gráfico de violino com os filtros selecionados.")

        st.markdown("---") # Separador para o próximo gráfico

        # --- Filtra o DataFrame de egressos com base nos cursos selecionados no filtro global ---
        df_egressos_filtered_by_course = filtered_egressos.copy()
        
        if selected_cursos and 'nome_curso' in df_egressos_filtered_by_course.columns:
            df_egressos_filtered_by_course = df_egressos_filtered_by_course[
                df_egressos_filtered_by_course['nome_curso'].isin(selected_cursos)
            ]
        elif not selected_cursos:
            st.info("Selecione um ou mais cursos no filtro 'Filtrar por Curso:' para visualizar os gráficos abaixo.")
            df_egressos_filtered_by_course = pd.DataFrame() # Esvazia o DF se nenhum curso for selecionado

        if not df_egressos_filtered_by_course.empty:
            
            # --- Gráfico de Barras de Contagem (Total de Alunos por Ano e Sexo nos Cursos Selecionados) ---
            st.subheader("Contagem de Egressos por Ano e Gênero nos Cursos Selecionados")
            
            if 'ano' in df_egressos_filtered_by_course.columns and \
               'sexo_rotulo' in df_egressos_filtered_by_course.columns:
                
                # Agrupa os dados para a contagem por ano e sexo nos cursos filtrados
                egressos_por_ano_sexo_curso = df_egressos_filtered_by_course.groupby(
                    ['ano', 'sexo_rotulo']
                ).size().reset_index(name='Contagem')

                fig_count_plot = px.bar(
                    egressos_por_ano_sexo_curso,
                    x='ano',
                    y='Contagem',
                    color='sexo_rotulo',
                    barmode='group', # Para barras agrupadas por sexo
                    title='Contagem de Egressos por Ano e Gênero nos Cursos Selecionados',
                    labels={'ano': 'Ano de Egresso', 'Contagem': 'Número de Egressos', 'sexo_rotulo': 'Gênero'},
                    color_discrete_sequence=px.colors.sequential.Magma # Paleta de cores 'magma'
                )
                fig_count_plot.update_xaxes(dtick=1, tickformat="%Y", tickangle=45)
                fig_count_plot.update_layout(yaxis_title='Número de Egressos')
                st.plotly_chart(fig_count_plot, use_container_width=True)
            else:
                st.info("Colunas 'ano' ou 'sexo_rotulo' não disponíveis nos dados filtrados para este gráfico.")


            # --- Histograma (Distribuição Geral de Períodos por Sexo nos Cursos Selecionados) ---
            st.subheader("Distribuição de Frequência do Total de Períodos nos Cursos Selecionados por Gênero")

            if 'total_periodos' in df_egressos_filtered_by_course.columns and \
               'sexo_rotulo' in df_egressos_filtered_by_course.columns:
                
                fig_hist_plot = px.histogram(
                    df_egressos_filtered_by_course,
                    x='total_periodos',
                    color='sexo_rotulo',
                    marginal="box", # Adiciona um box plot marginal para visualização da distribuição
                    nbins=LIMITE_MAX_PERIODOS, # Controla o número de bins, inspirado no 'bins' do seaborn
                    title=f'Distribuição de Frequência do Total de Períodos nos Cursos Selecionados (Máx {LIMITE_MAX_PERIODOS}) por Gênero',
                    labels={'total_periodos': 'Total de Semestres', 'sexo_rotulo': 'Gênero'},
                    color_discrete_sequence=px.colors.sequential.Cividis, # Paleta de cores 'cividis'
                    height=500
                )
                fig_hist_plot.update_layout(yaxis_title='Frequência')
                st.plotly_chart(fig_hist_plot, use_container_width=True)
            else:
                st.info("Colunas 'total_periodos' ou 'sexo_rotulo' não disponíveis nos dados filtrados para este gráfico.")

        elif selected_cursos: # if df_egressos_filtered_by_course is empty but selected_cursos is not
             st.info("Nenhum dado disponível para os cursos selecionados com os filtros atuais.")
        # else: A mensagem "Selecione um ou mais cursos..." já é exibida acima.


    else:
        st.info("Nenhum dado de egressos disponível com os filtros selecionados para análise.")


# --- TAB 3: Comparativo Geral ---
with tab_comparacao_viz:
    st.header("Comparativo Geral entre Ingressantes e Egressos")

    if not filtered_ingressantes.empty and not filtered_egressos.empty and \
       'ano' in filtered_ingressantes.columns and 'ano' in filtered_egressos.columns:

        with st.container():
            col_comp1, col_comp2 = st.columns(2)

            with col_comp1:
                st.subheader("Total de Ingressantes vs Egressos por Ano")
                ingressantes_count = filtered_ingressantes.groupby('ano').size().reset_index(name='Contagem')
                ingressantes_count['Tipo de Aluno'] = 'Ingressantes'

                egressos_count = filtered_egressos.groupby('ano').size().reset_index(name='Contagem')
                egressos_count['Tipo de Aluno'] = 'Egressos'

                combined_annual_data = pd.concat([ingressantes_count, egressos_count], ignore_index=True)

                fig_comp_ano = px.line(combined_annual_data, x='ano', y='Contagem', color='Tipo de Aluno',
                                       title='Total de Ingressantes vs Egressos por Ano',
                                       labels={'ano': 'Ano', 'Contagem': 'Número de Alunos'})
                fig_comp_ano.update_xaxes(dtick=1, tickformat="%Y")
                st.plotly_chart(fig_comp_ano, use_container_width=True)

            with col_comp2:
                st.subheader("Ingressantes e Egressos por Sexo ao Longo do Tempo")
                if 'sexo' in filtered_ingressantes.columns and 'sexo' in filtered_egressos.columns:
                    sex_ing_anual = filtered_ingressantes.groupby(['ano', 'sexo']).size().reset_index(name='count')
                    sex_ing_anual['Tipo'] = 'Ingressantes'

                    sex_eg_anual = filtered_egressos.groupby(['ano', 'sexo']).size().reset_index(name='count')
                    sex_eg_anual['Tipo'] = 'Egressos'

                    combined_sex_anual_data = pd.concat([sex_ing_anual, sex_eg_anual], ignore_index=True)

                    fig_comp_sex_time = px.bar(combined_sex_anual_data, x='ano', y='count', color='sexo',
                                               facet_col='Tipo', barmode='group',
                                               title='Ingressantes e Egressos por Sexo ao Longo do Tempo',
                                               labels={'ano': 'Ano', 'count': 'Número de Alunos', 'sexo': 'Sexo'})
                    fig_comp_sex_time.update_xaxes(dtick=1, tickformat="%Y")
                    st.plotly_chart(fig_comp_sex_time, use_container_width=True)
                else:
                    st.info("Coluna 'sexo' não disponível em um ou ambos os DataFrames para gráficos comparativos por sexo.")

    else:
        st.info("Dados incompletos ou insuficientes para a aba de comparação. Verifique os filtros selecionados e se há dados para ambos os grupos.")

# --- TAB 4: Machine Learning - Predição de Duração de Curso ---
# with tab_ml:
#     st.header("Predição do Tempo de Duração do Curso (Machine Learning)")
#     st.write("Esta seção utiliza um modelo de Machine Learning para prever se a duração do curso será 'Curta' ou 'Longa' para um aluno egresso, com base em características do curso e do aluno. **A unidade acadêmica não é utilizada nesta predição.**") # Texto atualizado

#     if filtered_egressos.empty or 'total_periodos' not in filtered_egressos.columns:
#         st.warning("Não há dados de egressos suficientes ou a coluna 'total_periodos' não está disponível para treinar o modelo de Machine Learning.")
#     else:
#         # 1. Preparação dos Dados para ML
#         ml_df = filtered_egressos.copy()

#         # Remover linhas com valores zero ou nulos em 'total_periodos' que não façam sentido para a predição
#         ml_df = ml_df[ml_df['total_periodos'] > 0]
#         # Remover 'nome_unidade' daqui também
#         ml_df.dropna(subset=['nivel_ensino', 'sexo', 'nome_curso', 'total_periodos'], inplace=True) # ATUALIZADO

#         if ml_df.empty:
#             st.warning("Após a limpeza, o DataFrame para Machine Learning está vazio. Certifique-se de que há dados de egressos válidos com 'total_periodos' > 0 e informações completas nas colunas de características.")
#         else:
#             # Definir a variável alvo (y) baseada na mediana de 'total_periodos'
#             median_periods = ml_df['total_periodos'].median()
#             st.info(f"A mediana de 'total_periodos' para este conjunto de dados é: **{median_periods:.2f}**.")
#             st.write(f"Consideramos 'Curto' se a duração for <= {int(median_periods)} períodos e 'Longo' se for > {int(median_periods)} períodos.")
            
#             ml_df['duracao_curso'] = ml_df['total_periodos'].apply(lambda x: 'Curto' if x <= median_periods else 'Longo')

#             # Features (X) e Target (y)
#             # 'nome_unidade' foi removido da lista de features
#             features = ['nivel_ensino', 'sexo', 'nome_curso'] # ATUALIZADO
#             target = 'duracao_curso'

#             X = ml_df[features]
#             y = ml_df[target]

#             # Codificação de variáveis categóricas (Label Encoding)
#             encoders = {}
#             X_encoded = X.copy()
#             for col in features:
#                 le = LabelEncoder()
#                 X_encoded[col] = le.fit_transform(X_encoded[col])
#                 encoders[col] = le

#             # Divisão dos dados em treino e teste
#             X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=y)

#             # Treinamento do Modelo (Decision Tree Classifier)
#             model = DecisionTreeClassifier(random_state=42)
#             model.fit(X_train, y_train)

#             # Avaliação do Modelo
#             y_pred = model.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
#             report = classification_report(y_test, y_pred, output_dict=True)

#             st.subheader("Métricas de Desempenho do Modelo")
#             st.write(f"**Acurácia do Modelo:** {accuracy:.2f}")
#             st.write("---")

#             st.write("**Relatório de Classificação:**")
#             st.json(report)
#             st.write("---")

#             st.subheader("Matriz de Confusão")
#             cm = confusion_matrix(y_test, y_pred, labels=['Curto', 'Longo'])
#             fig_cm = go.Figure(data=go.Heatmap(
#                 z=cm,
#                 x=['Previsto Curto', 'Previsto Longo'],
#                 y=['Real Curto', 'Real Longo'],
#                 colorscale='Viridis',
#                 colorbar=dict(title='Contagem')
#             ))
#             fig_cm.update_layout(title='Matriz de Confusão',
#                                  xaxis_title='Classe Predita',
#                                  yaxis_title='Classe Verdadeira')
#             st.plotly_chart(fig_cm, use_container_width=True)
#             st.write("---")

#             # 2. Interface de Predição para o Usuário
#             st.subheader("Faça uma Predição para um Novo Aluno Egresso")

#             # Obter opções únicas para os seletores de ML com base nos dados originais (ml_df)
#             ml_nivel_ensino_options = sorted(ml_df['nivel_ensino'].unique())
#             ml_sexo_options = sorted(ml_df['sexo'].unique())
#             ml_nome_curso_options = sorted(ml_df['nome_curso'].unique())
#             # ml_nome_unidade_options removido aqui

#             input_nivel_ensino = st.selectbox(
#                 "Nível de Ensino:", 
#                 options=ml_nivel_ensino_options,
#                 index=0 if ml_nivel_ensino_options else None
#             )
#             input_sexo = st.selectbox(
#                 "Sexo:", 
#                 options=ml_sexo_options,
#                 index=0 if ml_sexo_options else None
#             )
#             input_nome_curso = st.selectbox(
#                 "Nome do Curso:", 
#                 options=ml_nome_curso_options,
#                 index=0 if ml_nome_curso_options else None
#             )
#             # Removido o seletor de unidade acadêmica
#             # input_nome_unidade = st.selectbox(
#             #     "Nome da Unidade:", 
#             #     options=ml_nome_unidade_options,
#             #     index=0 if ml_nome_unidade_options else None
#             # )

#             predict_button = st.button("Prever Duração do Curso")

#             if predict_button:
#                 # Ajustada a verificação para as features restantes
#                 if None in [input_nivel_ensino, input_sexo, input_nome_curso]: # ATUALIZADO
#                     st.error("Por favor, selecione todas as opções para fazer a predição.")
#                 else:
#                     try:
#                         # Codificar as entradas do usuário
#                         encoded_nivel_ensino = encoders['nivel_ensino'].transform([input_nivel_ensino])[0]
#                         encoded_sexo = encoders['sexo'].transform([input_sexo])[0]
#                         encoded_nome_curso = encoders['nome_curso'].transform([input_nome_curso])[0]
#                         # encoded_nome_unidade removido

#                         # Criar DataFrame para a predição
#                         # Removido 'encoded_nome_unidade' dos dados de entrada
#                         input_data = pd.DataFrame([[encoded_nivel_ensino, encoded_sexo, encoded_nome_curso]], # ATUALIZADO
#                                                     columns=X_encoded.columns)

#                         # Fazer a predição
#                         prediction = model.predict(input_data)[0]
#                         prediction_proba = model.predict_proba(input_data)[0]

#                         st.success(f"A predição para este aluno é: **{prediction}**")
#                         # Garantir que a ordem das classes seja consistente
#                         proba_curto = prediction_proba[np.where(model.classes_ == 'Curto')[0][0]] if 'Curto' in model.classes_ else 0
#                         proba_longo = prediction_proba[np.where(model.classes_ == 'Longo')[0][0]] if 'Longo' in model.classes_ else 0

#                         st.info(f"Probabilidade de ser 'Curto': **{proba_curto:.2f}**")
#                         st.info(f"Probabilidade de ser 'Longo': **{proba_longo:.2f}**")

#                     except ValueError as ve:
#                         st.error(f"Erro ao codificar a entrada. Uma das opções selecionadas pode não ter sido vista durante o treinamento do modelo. Por favor, tente novamente. Detalhes: {ve}")
#                     except Exception as e:
#                         st.error(f"Ocorreu um erro ao fazer a predição: {e}")

# --- NOVA TAB PARA O MODELO DE CLASSIFICAÇÃO ---
with tab_ml_classificacao:
    st.header("🎯 Predição de Desempenho (Classificação)")
    st.write("Preveja se um aluno terá um tempo de graduação 'Curto' ou 'Longo' baseado em suas características. Este modelo usa a coluna `total_periodos` para classificar os egressos.")

    # --- Verificação de dados para o modelo de classificação ---
    if filtered_egressos.empty or 'total_periodos' not in filtered_egressos.columns:
        st.warning("Não há dados de egressos com 'total_periodos' para treinar o modelo de classificação com os filtros atuais. Ajuste os filtros ou verifique seus dados de egressos.")
        st.stop()

    # Definição do limiar para "Curto" vs "Longo"
    st.subheader("Definição do Limiar e Preparação dos Dados")
    st.info("O tempo médio de graduação dos dados filtrados é de aproximadamente "
            f"**{filtered_egressos['total_periodos'].mean():.1f} períodos**.")

    # Limiar dinâmico ou fixo
    median_periods = filtered_egressos['total_periodos'].median() if not filtered_egressos['total_periodos'].empty else 8 # Valor padrão razoável
    threshold = st.slider(
        "Defina o limiar para 'Curto' (em períodos, menor ou igual a este valor)",
        min_value=1, max_value=24, value=int(median_periods), step=1
    )
    st.write(f"Alunos com tempo de graduação <= {threshold} períodos serão classificados como 'Curto'.")

    # Criação da variável alvo
    df_model = filtered_egressos.copy()
    df_model['desempenho'] = df_model['total_periodos'].apply(lambda x: 'Curto' if x <= threshold else 'Longo')

    # Contagem das classes
    class_counts = df_model['desempenho'].value_counts()
    st.info(f"Distribuição das classes: Curto = {class_counts.get('Curto', 0)}, Longo = {class_counts.get('Longo', 0)}")

    # Seleção de Features para o modelo de Classificação
    # Ajuste essas features com base na relevância do seu dataset
    features_classificacao = ['nivel_ensino', 'sexo', 'nome_curso', 'nome_unidade']
    target_classificacao = 'desempenho'

    # Verifica se as colunas selecionadas existem no DataFrame
    missing_features = [f for f in features_classificacao if f not in df_model.columns]
    if missing_features:
        st.error(f"As seguintes features estão faltando no DataFrame de egressos: {', '.join(missing_features)}. Ajuste a seleção de features ou verifique seus dados.")
        st.stop()

    X = df_model[features_classificacao]
    y = df_model[target_classificacao]

    # Codificação de variáveis categóricas
    # Usaremos LabelEncoder para cada coluna individualmente e armazenaremos os encoders
    encoders = {}
    X_encoded_df = pd.DataFrame()
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded_df[col] = le.fit_transform(X[col])
            encoders[col] = le
        else:
            X_encoded_df[col] = X[col]

    # Divisão em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y, test_size=0.2, random_state=42, stratify=y)

    # Treinamento do Modelo de Classificação
    @st.cache_resource
    def train_classification_model(X_train, y_train):
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model

    model = train_classification_model(X_train, y_train)

    # Avaliação do Modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Modelo de Classificação treinado com sucesso! 🎉")
    st.info(f"Acurácia no conjunto de teste: **{accuracy:.2f}**")
    st.text("Relatório de Classificação:")
    st.code(classification_report(y_test, y_pred))
    st.text("Matriz de Confusão:")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred), index=model.classes_, columns=model.classes_))

    st.markdown("---")
    st.subheader("Simulação de Predição de Desempenho")
    st.write("Insira as características de um aluno para prever se ele terá um tempo de graduação 'Curto' ou 'Longo'.")

    # Coletar inputs do usuário para as features de classificação
    input_values_class = {}
    for feature in features_classificacao:
        if feature in encoders: # Variáveis categóricas
            unique_options = sorted(encoders[feature].classes_.tolist())
            input_values_class[feature] = st.selectbox(f"Selecione o(a) {feature.replace('_', ' ').title()}", options=unique_options, key=f"class_{feature}")
        else: # Variáveis numéricas (se houver)
            st.write(f"Feature numérica '{feature}' não é suportada diretamente neste formulário de classificação. Considerar tratamento.")

    if st.button("Prever Desempenho"):
        try:
            # Criar DataFrame com os inputs do usuário para classificação
            input_data_class = pd.DataFrame([input_values_class])

            # Codificar a entrada do usuário usando os encoders treinados
            input_encoded_dict = {}
            for col in X.columns:
                if col in encoders:
                    # Usar .transform para garantir que novas categorias causem erro e não NaN
                    # Adicionado erro handling para categorias desconhecidas
                    try:
                        input_encoded_dict[col] = encoders[col].transform(input_data_class[col])
                    except ValueError as ve:
                        st.error(f"Erro ao codificar a feature '{col}': A opção '{input_data_class[col].iloc[0]}' não foi vista durante o treinamento do modelo. Por favor, selecione uma opção válida.")
                        st.stop()
                else:
                    input_encoded_dict[col] = input_data_class[col] # Para colunas numéricas, se houver

            input_data_encoded = pd.DataFrame(input_encoded_dict, index=[0])

            # Garantir que as colunas e a ordem sejam as mesmas usadas no treinamento
            input_data_aligned = input_data_encoded.reindex(columns=X_train.columns, fill_value=0)

            # Fazer a predição
            prediction = model.predict(input_data_aligned)[0]
            prediction_proba = model.predict_proba(input_data_aligned)[0]

            st.success(f"A predição para este aluno é: **{prediction}**")
            # Garantir que a ordem das classes seja consistente
            proba_curto = prediction_proba[np.where(model.classes_ == 'Curto')[0][0]] if 'Curto' in model.classes_ else 0
            proba_longo = prediction_proba[np.where(model.classes_ == 'Longo')[0][0]] if 'Longo' in model.classes_ else 0

            st.info(f"Probabilidade de ser 'Curto': **{proba_curto:.2f}**")
            st.info(f"Probabilidade de ser 'Longo': **{proba_longo:.2f}**")

        except Exception as e:
            st.error(f"Ocorreu um erro ao fazer a predição: {e}")
            st.write("Detalhes do erro:", e)


# --- NOVA TAB PARA O MODELO DE REGRESSÃO ---
with tab_ml_regressao:
    st.header("⏳ Predição do Tempo de Graduação (Regressão)")
    st.write("Utilize este modelo para estimar o número de períodos necessários para a graduação de um aluno, com base em suas características.")

    # --- Verificação de dados para o modelo de regressão ---
    # O modelo de regressão precisa de 'total_periodos' que está em egressos
    if filtered_egressos.empty or 'total_periodos' not in filtered_egressos.columns or filtered_egressos['total_periodos'].isnull().all():
        st.warning("Não há dados de egressos com 'total_periodos' válidos para treinar o modelo de regressão com os filtros atuais. Ajuste os filtros ou verifique seus dados de egressos.")
        st.stop()

    # --- Preparação dos dados para o Modelo de Regressão ---
    st.subheader("Configuração e Treinamento do Modelo")

    # Features para o modelo de regressão (ajuste conforme a relevância do seu dataset)
    features_regressao = ['nivel_ensino', 'sexo', 'nome_curso', 'nome_unidade', 'ano_ingresso']
    target_regressao = 'total_periodos'

    # Filtrar o DataFrame de egressos para incluir apenas as colunas relevantes e remover NaNs no target
    df_regressao = filtered_egressos[features_regressao + [target_regressao]].dropna(subset=[target_regressao]).copy()

    if df_regressao.empty:
        st.warning("Após a seleção de features e remoção de valores ausentes, o DataFrame de regressão está vazio. O modelo não pode ser treinado.")
        st.stop()

    # Certificar-se de que 'ano_ingresso' é numérica
    if 'ano_ingresso' in df_regressao.columns:
        df_regressao['ano_ingresso'] = pd.to_numeric(df_regressao['ano_ingresso'], errors='coerce').fillna(df_regressao['ano_ingresso'].mean())

    # Separação de features (X) e target (y)
    X_reg = df_regressao[features_regressao]
    y_reg = df_regressao[target_regressao]

    # Codificação de variáveis categóricas para o modelo de regressão
    # Usar pd.get_dummies é mais robusto para novas categorias em inferência
    X_reg_encoded = pd.get_dummies(X_reg, columns=[col for col in features_regressao if X_reg[col].dtype == 'object'])

    # Divisão em conjuntos de treino e teste
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg_encoded, y_reg, test_size=0.2, random_state=42)

    # Treinamento do Modelo de Regressão (RandomForestRegressor como exemplo)
    @st.cache_resource
    def train_regression_model(X_train, y_train):
        model_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 para usar todos os cores
        model_reg.fit(X_train, y_train)
        return model_reg

    model_reg = train_regression_model(X_train_reg, y_train_reg)

    # Avaliação do Modelo
    y_pred_reg = model_reg.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse) # Root Mean Squared Error
    r2 = r2_score(y_test_reg, y_pred_reg)

    st.success(f"Modelo de Regressão treinado com sucesso! 🎉")
    st.info(f"Métricas de Avaliação no conjunto de teste:")
    st.markdown(f"- Erro Quadrático Médio (MSE): **{mse:.2f}**")
    st.markdown(f"- Raiz do Erro Quadrático Médio (RMSE): **{rmse:.2f}**")
    st.markdown(f"- Coeficiente de Determinação (R² Score): **{r2:.2f}**")
    st.markdown("Um R² mais próximo de 1 indica um modelo que explica melhor a variância dos dados.")

    st.markdown("---")
    st.subheader("Simulação de Predição de Períodos")
    st.write("Insira as características de um aluno para prever o número de períodos até a graduação.")

    # Coletar inputs do usuário para as features de regressão
    input_reg_values = {}
    for feature in features_regressao:
        if feature == 'ano_ingresso':
            # Obter o intervalo de anos válidos dos dados para o slider/number_input
            min_year_input = int(X_reg['ano_ingresso'].min()) if not X_reg['ano_ingresso'].empty else 2014
            max_year_input = int(X_reg['ano_ingresso'].max()) if not X_reg['ano_ingresso'].empty else 2024
            input_reg_values[feature] = st.number_input(
                f"Ano de Ingresso",
                min_value=min_year_input,
                max_value=max_year_input,
                value=int(X_reg['ano_ingresso'].mode()[0]) if not X_reg['ano_ingresso'].empty else 2020,
                step=1,
                key=f"reg_input_{feature}" # Chave única para o widget
            )
        elif X_reg[feature].dtype == 'object':
            unique_options = sorted(filtered_egressos[feature].unique().tolist())
            input_reg_values[feature] = st.selectbox(f"Selecione o(a) {feature.replace('_', ' ').title()}", options=unique_options, key=f"reg_input_{feature}")
        else: # Para outras features numéricas, se houver
            input_reg_values[feature] = st.number_input(f"Valor para '{feature.replace('_', ' ').title()}'", value=float(X_reg[feature].mean()), key=f"reg_input_{feature}")

    if st.button("Prever Duração da Graduação", key="predict_reg_button"): # Chave única para o botão
        try:
            # Criar DataFrame com os inputs do usuário
            input_data_reg = pd.DataFrame([input_reg_values])

            # Codificar variáveis categóricas do input, garantindo que as colunas sejam as mesmas do treino
            input_data_reg_encoded = pd.get_dummies(input_data_reg, columns=[col for col in features_regressao if input_data_reg[col].dtype == 'object'])

            # Alinhar colunas do input com as colunas usadas no treinamento do modelo (importante para get_dummies)
            # Adicionar colunas ausentes e preencher com 0
            missing_cols_input = set(X_train_reg.columns) - set(input_data_reg_encoded.columns)
            for c in missing_cols_input:
                input_data_reg_encoded[c] = 0

            # Remover colunas extras do input que não foram vistas no treino
            extra_cols_input = set(input_data_reg_encoded.columns) - set(X_train_reg.columns)
            input_data_reg_encoded = input_data_reg_encoded.drop(columns=list(extra_cols_input), errors='ignore')

            # Reordenar colunas para que correspondam à ordem de treino
            input_data_reg_encoded = input_data_reg_encoded[X_train_reg.columns]

            # Fazer a predição
            predicted_periods = model_reg.predict(input_data_reg_encoded)[0]

            st.success(f"A duração estimada da graduação é de aproximadamente **{predicted_periods:.1f} períodos**.")
            st.info(f"Isso equivale a cerca de **{predicted_periods / 2:.1f} anos**.")

        except Exception as e:
            st.error(f"Ocorreu um erro ao fazer a predição: {e}. Verifique se todos os campos foram preenchidos corretamente.")
            st.write("Detalhes do erro:", e)


st.sidebar.markdown("---")
st.sidebar.info(
    "Este dashboard interativo permite explorar dados de ingressantes e egressos, "
    "analisando tendências e distribuições por ano, nível de ensino, sexo, curso e unidade."
    "\nCriado por: Robson Carneiro"
)