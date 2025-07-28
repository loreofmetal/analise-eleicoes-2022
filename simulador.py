import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Preditor de Votação - Eleições 2022 - 1o Turno",
    page_icon="🗳️",
    layout="wide"
)

# Title and description
st.title("🗳️ Preditor de Votação em Lula 2022")
st.markdown("""
**Objetivo**: Comparar três modelos de predição usando dados municipais brasileiros.
""")

@st.cache_data
def load_models():
    """Load all three models"""
    models = {}
    model_files = {
        'inicial': 'data/models/modelo_inicial_bolsa_familia.pkl',
        'regional': 'data/models/regional_model_bolsa_familia.pkl',
        'estados': 'data/models/model_states_fixed.pkl'
    }
    
    for model_name, file_path in model_files.items():
        try:
            with open(file_path, 'rb') as f:
                models[model_name] = pickle.load(f)
            #st.success(f"✅ {model_name.title()} model loaded successfully")
        except FileNotFoundError:
            st.warning(f"⚠️ Model file not found: {file_path}")
        except Exception as e:
            st.error(f"❌ Error loading {model_name} model: {e}")
            st.info(f"Skipping {model_name} model due to compatibility issues")
    
    if not models:
        st.error("No models could be loaded. Please check the model files.")
        return None
    
    return models

@st.cache_data
def load_data():
    """Load the merged dataframe for reference values"""
    try:
        df = pd.read_pickle('data/merged_df.pkl')
        return df
    except FileNotFoundError:
        st.error("Arquivo merged_df.pkl não encontrado. Execute primeiro os notebooks de análise.")
        return None

# Load models and data
models = load_models()
df = load_data()

# Load models and data
models = load_models()
df = load_data()

if models is None or not models:
    st.error("❌ Nenhum modelo pôde ser carregado.")
    st.stop()

# Create 3-column layout for more efficient use of space
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.subheader("📊 Entrada")
    
    # Bolsa Família percentage - main input
    perc_bolsa_familia = st.slider(
        "% Bolsa Família",
        min_value=0.0,
        max_value=30.0,
        value=15.0,
        step=0.5,
        help="Principal variável preditiva"
    )
    
    # Region and state in compact form
    regiao = st.selectbox(
        "Região",
        options=['Norte', 'Nordeste', 'Sudeste', 'Sul', 'Centro-Oeste'],
        index=1
    )
    
    # State selection
    estados_por_regiao = {
        'Norte': ['RO', 'AC', 'AM', 'RR', 'PA', 'AP', 'TO'],
        'Nordeste': ['MA', 'PI', 'CE', 'RN', 'PB', 'PE', 'AL', 'SE', 'BA'],
        'Sudeste': ['MG', 'ES', 'RJ', 'SP'],
        'Sul': ['PR', 'SC', 'RS'],
        'Centro-Oeste': ['MS', 'MT', 'GO', 'DF']
    }
    
    uf = st.selectbox("Estado", options=estados_por_regiao[regiao])

with col2:
    st.subheader("🎯 Predições")
    
    # Prepare input data for predictions
    def make_predictions():
        predictions = {}
        
        try:
            # Model 1: Initial model - only % Bolsa Família
            if 'inicial' in models:
                X_inicial = np.array([[perc_bolsa_familia]])
                pred_inicial = models['inicial'].predict(X_inicial)[0]
                predictions['Modelo 1'] = pred_inicial
            
            # Model 2: Regional model - % Bolsa Família + Nordeste dummy
            if 'regional' in models:
                is_nordeste = 1 if regiao == 'Nordeste' else 0
                X_regional = np.array([[perc_bolsa_familia, is_nordeste]])
                pred_regional = models['regional'].predict(X_regional)[0]
                predictions['Modelo 2'] = pred_regional
            
            # Model 3: State model - % Bolsa Família + state dummies
            if 'estados' in models:
                # AC is the reference state (first alphabetically), not MG
                # States with dummies (all except AC)
                states_with_dummies = ['AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 
                                      'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 
                                      'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO']
                
                # Create dummy variables (1 for selected state, 0 for others)
                # If the selected state is AC, all dummies will be 0 (reference category)
                if uf == 'AC':
                    state_dummies = [0] * 26  # All zeros for reference state (AC)
                else:
                    state_dummies = [1 if state == uf else 0 for state in states_with_dummies]
                
                X_estados = np.array([[perc_bolsa_familia] + state_dummies])
                pred_estados = models['estados'].predict(X_estados)[0]
                predictions['Modelo 3'] = pred_estados
                
        except Exception as e:
            st.error(f"Erro: {e}")
            return None
            
        return predictions
    
    # Make predictions
    predictions = make_predictions()
    
    if predictions:
        # Compact prediction display
        for i, (model_name, prediction) in enumerate(predictions.items()):
            color = ["#FF6B6B", "#4ECDC4", "#45B7D1"][i]
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, {color}20, {color}10); 
                        padding: 15px; margin: 8px 0; border-radius: 8px; 
                        border-left: 4px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold; color: {color};">{model_name}</span>
                    <span style="font-size: 20px; font-weight: bold;">{prediction:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Compact statistics
        avg_pred = np.mean(list(predictions.values()))
        range_pred = max(predictions.values()) - min(predictions.values())
        
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
            <small>
            <strong style="color: #212529;">Média:</strong>
            <span style="color: #212529;">{avg_pred:.1f}%</span>
            |
            <strong style="color: #212529;">Amplitude:</strong>
            <span style="color: #212529;">{range_pred:.1f}%</span>
            </small>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.subheader("ℹ️ Info")
    
    # Model evolution explanation
    with st.expander("🔬 Evolução dos Modelos", expanded=False):
        st.markdown("""
        **Por que 3 modelos diferentes?**
        
        **Modelo 1 - Baseline**  
        Testamos se apenas o % Bolsa Família consegue prever o voto em Lula. Resultado: correlação forte, mas limitada.
        
        **Modelo 2 - Efeito Regional**  
        Adicionamos o fator "região Nordeste", pois historicamente essa região vota mais no PT. Melhoria significativa na predição.
        
        **Modelo 3 - Efeitos Estaduais**  
        Incluímos fatores específicos de cada estado, capturando diferenças políticas locais que vão além da região. Máxima precisão alcançada.
        
        **Interpretação:**  
        Compare as predições para entender como fatores regionais e estaduais influenciam além da política social.
        """)
    
    # Compact model explanation
    st.markdown("""
    **Modelos:**
    - **1**: Só Bolsa Família
    - **2**: + Efeito Nordeste  
    - **3**: + Efeito Estadual
    """)
    
    # Compact data statistics
    if df is not None:
        st.markdown(f"""
        **Dados:**
        - Municípios: {len(df):,}
        - Voto Lula Mediano: {df['voto_lula'].median():.1f}%
        - Bolsa Família Mediano: {df['perc_bolsa_familia'].median():.1f}%
        """)

# Remove the old sidebar
# Add footer
st.markdown("---")
st.markdown("**Análise Eleições 2022** | Desenvolvido com Streamlit")