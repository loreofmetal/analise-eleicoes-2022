import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Preditor de Votação - Eleições 2022",
    page_icon="🗳️",
    layout="wide"
)

# Title and description
st.title("🗳️ Preditor de Votação em Lula 2022")
st.markdown("""
Este aplicativo compara três modelos de predição do percentual de votos em Lula no primeiro turno de 2022:
- **Modelo 1**: Baseado apenas no % de famílias beneficiárias do Bolsa Família
- **Modelo 2**: Adiciona efeito regional (Nordeste vs. outras regiões)  
- **Modelo 3**: Considera efeitos específicos de cada estado brasileiro
""")

st.info("💡 **Objetivo**: Analisar como variáveis socioeconômicas e geográficas influenciam o comportamento eleitoral")

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

if models is None:
    st.error("Could not load any models. Please check the model files.")
    st.stop()
elif not models:
    st.error("No models were successfully loaded.")
    st.stop()

if df is None:
    st.warning("Could not load data file. Some statistics will not be available.")

# Create columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📊 Dados de Entrada")
    
    # Bolsa Família percentage - main input for all models
    perc_bolsa_familia = st.slider(
        "% Famílias Bolsa Família",
        min_value=0.0,
        max_value=30.0,
        value=15.0,
        step=0.5,
        help="Percentual de famílias beneficiárias do Bolsa Família (principal variável preditiva)"
    )
    
    st.markdown("---")
    st.subheader("Para modelos regionais/estaduais:")
    
    # Region selection (for regional model)
    regiao = st.selectbox(
        "Região",
        options=['Norte', 'Nordeste', 'Sudeste', 'Sul', 'Centro-Oeste'],
        index=1,  # Default to Nordeste
        help="Região do município (para modelo regional)"
    )
    
    # State selection (for state model)
    estados_por_regiao = {
        'Norte': ['RO', 'AC', 'AM', 'RR', 'PA', 'AP', 'TO'],
        'Nordeste': ['MA', 'PI', 'CE', 'RN', 'PB', 'PE', 'AL', 'SE', 'BA'],
        'Sudeste': ['MG', 'ES', 'RJ', 'SP'],
        'Sul': ['PR', 'SC', 'RS'],
        'Centro-Oeste': ['MS', 'MT', 'GO', 'DF']
    }
    
    uf = st.selectbox(
        "Estado (UF)",
        options=estados_por_regiao[regiao],
        help="Estado do município (para modelo por estados)"
    )
    
    # Show which variables each model uses
    st.markdown("---")
    st.subheader("📋 Variáveis por modelo:")
    st.markdown("""
    **Modelo 1**: Apenas % Bolsa Família  
    **Modelo 2**: % Bolsa Família + Nordeste (sim/não)  
    **Modelo 3**: % Bolsa Família + Estado específico
    """)

with col2:
    st.header("🎯 Predições dos Modelos")
    
    # Prepare input data for predictions
    def make_predictions():
        predictions = {}
        
        try:
            # Model 1: Initial model - only % Bolsa Família
            if 'inicial' in models:
                X_inicial = np.array([[perc_bolsa_familia]])
                pred_inicial = models['inicial'].predict(X_inicial)[0]
                predictions['Modelo 1: Só Bolsa Família'] = pred_inicial
            
            # Model 2: Regional model - % Bolsa Família + Nordeste dummy
            if 'regional' in models:
                # Create Nordeste dummy (1 if Nordeste, 0 otherwise)
                is_nordeste = 1 if regiao == 'Nordeste' else 0
                X_regional = np.array([[perc_bolsa_familia, is_nordeste]])
                pred_regional = models['regional'].predict(X_regional)[0]
                predictions['Modelo 2: Bolsa Família + Nordeste'] = pred_regional
            
            # Model 3: State model - % Bolsa Família + state dummies
            if 'estados' in models:
                # Create state dummies for 26 states (excluding MG which is the reference)
                all_states = ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 
                             'MA', 'MT', 'MS', 'PA', 'PB', 'PR', 'PE', 'PI', 
                             'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SE', 'SP', 'TO']
                # Note: MG is excluded as it's the reference state
                
                # Create dummy variables (1 for selected state, 0 for others)
                # If the selected state is MG, all dummies will be 0 (reference category)
                if uf == 'MG':
                    state_dummies = [0] * 26  # All zeros for reference state
                else:
                    state_dummies = [1 if state == uf else 0 for state in all_states]
                
                # Combine Bolsa Família with state dummies (should be 27 features total)
                X_estados = np.array([[perc_bolsa_familia] + state_dummies])
                pred_estados = models['estados'].predict(X_estados)[0]
                predictions['Modelo 3: Bolsa Família + Estado'] = pred_estados
            
            if not predictions:
                st.error("No models available for predictions")
                return None
                
        except Exception as e:
            st.error(f"Erro ao fazer predições: {e}")
            st.info("Tip: Try different input values or check model compatibility")
            return None
            
        return predictions
    
    # Make predictions
    predictions = make_predictions()
    
    if predictions:
        # Display predictions in cards
        for model_name, prediction in predictions.items():
            with st.container():
                st.markdown(f"""
                <div style="
                    background-color: #f0f2f6;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                    border-left: 5px solid #1f77b4;
                ">
                    <h3 style="margin: 0; color: #1f77b4;">{model_name}</h3>
                    <h2 style="margin: 10px 0; color: #333;">{prediction:.2f}%</h2>
                    <p style="margin: 0; color: #666;">Predição de votos em Lula</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Show comparison
        st.subheader("📈 Comparação dos Modelos")
        
        pred_df = pd.DataFrame({
            'Modelo': list(predictions.keys()),
            'Predição (%)': list(predictions.values())
        })
        
        st.bar_chart(pred_df.set_index('Modelo'))
        
        # Show statistics
        avg_pred = np.mean(list(predictions.values()))
        std_pred = np.std(list(predictions.values()))
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            st.metric("Média", f"{avg_pred:.2f}%")
        
        with col_stats2:
            st.metric("Desvio Padrão", f"{std_pred:.2f}%")
            
        with col_stats3:
            st.metric("Amplitude", f"{max(predictions.values()) - min(predictions.values()):.2f}%")

# Add sidebar with information
with st.sidebar:
    st.header("ℹ️ Sobre os Modelos")
    
    st.markdown("""
    **Modelo 1**: Apenas % Bolsa Família
    
    **Modelo 2**: % Bolsa Família + Se é Nordeste (sim/não)
    
    **Modelo 3**: % Bolsa Família + Estado específico (UF)
    """)
    
    if df is not None:
        st.header("📊 Estatísticas dos Dados")
        st.metric("Municípios", f"{len(df):,}")
        st.metric("% Voto Lula Mediano", f"{df['voto_lula'].median():.1f}%")
        st.metric("% Bolsa Família Mediano", f"{df['perc_bolsa_familia'].median():.1f}%")

    st.header("🚀 Como usar")
    st.markdown("""
    1. Ajuste o % de Bolsa Família (principal variável)
    2. Selecione a região e estado para modelos 2 e 3
    3. Compare como cada modelo responde
    4. Observe o efeito regional vs. estadual
    """)

    st.header("🎯 Interpretação")
    st.markdown("""
    - **Modelo 1**: Efeito "puro" do Bolsa Família
    - **Modelo 2**: Adiciona efeito específico do Nordeste
    - **Modelo 3**: Considera particularidades de cada estado
    """)

# Footer
st.markdown("---")
st.markdown("**Análise Eleições 2022** | Desenvolvido com Streamlit")