# 🗳️ Preditor de Votação - Eleições 2022

Aplicação interativa para predição de votação em Lula nas eleições presidenciais de 2022, usando dados municipais brasileiros.

## 📊 Modelos

- **Modelo 1**: Predição baseada apenas no % Bolsa Família
- **Modelo 2**: Adiciona efeito regional (Nordeste)  
- **Modelo 3**: Adiciona efeitos específicos por estado

## 🚀 Como usar

### Online (Recomendado)
Acesse a aplicação hospedada: [Em breve]

### Local
```bash
pip install -r requirements.txt
streamlit run simulador.py
```

## 📈 Dados
- **Fonte**: TSE (resultados eleitorais) + dados socioeconômicos municipais
- **Período**: Eleições 2022 (1º turno)
- **Cobertura**: Todos os municípios brasileiros

## 🔬 Metodologia
Regressão linear com variáveis progressivamente mais complexas para demonstrar a importância de fatores regionais e estaduais na predição eleitoral.

---
**Desenvolvido com Streamlit** 🎈
