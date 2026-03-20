import streamlit as st
import pandas as pd
import joblib

# Configuração da página (deve ser a primeira linha)
st.set_page_config(page_title="Marketing Cluster", page_icon="🎯")

# Carregamento dos modelos
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')

# --- SIDEBAR (INFORMAÇÕES TÉCNICAS) ---
st.sidebar.header("Sobre o Projeto")
st.sidebar.write("""
Este projeto utiliza **K-Means Clustering** para identificar padrões de comportamento.
Ao segmentar o público em 'bolhas' de interesse, permitimos campanhas mais assertivas.
""")

st.sidebar.subheader("Definição dos Grupos:")
st.sidebar.info("""
- **Grupo 0:** Público jovem (Moda, Música, Aparência).
- **Grupo 1:** Esportes e Cultura (Futebol, Basquete, Rock).
- **Grupo 2:** Equilibrado (Dança, Música, Estilo de vida).
""")

# --- BOTÃO DE DOWNLOAD DO EXEMPLO ---
st.sidebar.markdown("---") # Cria uma linha divisória
st.sidebar.subheader("Precisa de um modelo?")

# Abrimos o arquivo que já está na sua pasta
with open("Grupos_interesse.csv", "rb") as file:
    btn = st.sidebar.download_button(
        label="📥 Baixar arquivo de exemplo",
        data=file,
        file_name="exemplo_para_teste.csv",
        mime="text/csv"
    )

st.sidebar.caption("Use este arquivo para testar o formato aceito pelo modelo.")

# --- CORPO PRINCIPAL ---
st.title('🎯 Grupos de Interesse para Marketing')

st.write("Escolha um arquivo CSV para realizar a segmentação automática dos usuários.")

up_file = st.file_uploader('Upload do arquivo CSV', type='csv')

def processar_prever(df):
    # CORREÇÃO DO ERRO: Mantém apenas as colunas que o modelo conhece
    # Se o arquivo já tiver a coluna 'grupos', ela é ignorada aqui
    colunas_necessarias = ['sexo'] + [c for c in df.columns if c not in ['sexo', 'grupos']]
    dados_filtrados = df[colunas_necessarias].copy()

    # Processamento do Sexo
    encoded_sexo = encoder.transform(dados_filtrados[['sexo']])
    encoded_df = pd.DataFrame(encoded_sexo, columns=encoder.get_feature_names_out(['sexo']))
    
    # Concatenar e remover coluna original de sexo
    dados_final = pd.concat([dados_filtrados.drop('sexo', axis=1).reset_index(drop=True), 
                             encoded_df.reset_index(drop=True)], axis=1)

    # Escalonamento e Predição
    dados_escalados = scaler.transform(dados_final)
    clusters = kmeans.predict(dados_escalados)
    return clusters

if up_file is not None:
    df_original = pd.read_csv(up_file)
    
    with st.spinner('Processando inteligência artificial...'):
        # Executa a predição
        res_clusters = processar_prever(df_original)
        
        # Cria uma cópia para exibir resultados
        df_resultado = df_original.copy()
        if 'grupos' in df_resultado.columns:
            df_resultado = df_resultado.drop(columns=['grupos'])
        
        df_resultado.insert(0, 'grupos', res_clusters)

    # --- EXIBIÇÃO DE RESULTADOS ---
    st.success('Análise concluída com sucesso!')
    
    # Métricas Rápidas
    col1, col2, col3 = st.columns(3)
    contagem = df_resultado['grupos'].value_counts()
    col1.metric("Total de Usuários", len(df_resultado))
    col2.metric("Grupo mais frequente", contagem.idxmax())
    col3.metric("Qtd Grupos", len(contagem))

    st.subheader('Amostra dos Resultados (Top 10)')
    st.dataframe(df_resultado.head(10), use_container_width=True)

    # Download
    csv_data = df_resultado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='📥 Baixar Relatório Completo (CSV)',
        data=csv_data,
        file_name='predicao_marketing.csv',
        mime='text/csv'
    )