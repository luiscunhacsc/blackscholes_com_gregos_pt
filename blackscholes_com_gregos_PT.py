import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Função Black-Scholes com cálculo dos Greeks
def black_scholes_greeks(S, K, T, r, sigma, tipo_opcao='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if tipo_opcao == 'call':
        preço = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        preço = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    # Greeks comuns a call e put
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * norm.cdf(d2 if tipo_opcao == 'call' else -d2))
    
    return preço, delta, gamma, theta, vega, rho

# Configuração da interface
st.set_page_config(layout="wide")
st.title("📊 Compreender os **Greeks** no Modelo Black-Scholes")
st.markdown("Explore como os parâmetros afetam a sensibilidade do preço das opções (Delta, Gamma, Theta, Vega, Rho).")

# Sidebar com parâmetros
with st.sidebar:
    st.header("⚙️ Parâmetros")
    S = st.slider("Preço Atual do Ativo (S)", 50.0, 150.0, 100.0)
    K = st.slider("Preço de Exercício (K)", 50.0, 150.0, 105.0)
    T = st.slider("Tempo até Vencimento (anos)", 0.1, 5.0, 1.0)
    r = st.slider("Taxa de Juro Sem Risco (r)", 0.0, 0.2, 0.05)
    sigma = st.slider("Volatilidade (σ)", 0.1, 1.0, 0.2)
    tipo_opcao = st.radio("Tipo de Opção", ["call", "put"])

# Calcular preço e Greeks
preço, delta, gamma, theta, vega, rho = black_scholes_greeks(S, K, T, r, sigma, tipo_opcao)

# Mostrar resultados em colunas
col1, col2 = st.columns([1, 3])
with col1:
    st.success(f"### Preço da Opção: **€{preço:.2f}**")
    
    # Tabela de Greeks
    st.markdown("### Sensibilidades (Greeks)")
    st.markdown(f"""
    - **Delta (Δ):** `{delta:.3f}`  
      *Mudança no preço da opção por €1 no ativo.*
    - **Gamma (Γ):** `{gamma:.3f}`  
      *Mudança no Delta por €1 no ativo.*
    - **Theta (Θ):** `{theta:.3f}/dia`  
      *Erosão do valor com o tempo (por dia).*
    - **Vega (ν):** `{vega:.3f}`  
      *Mudança por 1% de volatilidade.*
    - **Rho (ρ):** `{rho:.3f}`  
      *Impacto de 1% na taxa de juro.*
    """)

with col2:
    # Selecionar Greek para visualizar
    greek_selecionado = st.selectbox(
        "Escolha um Greek para visualizar:",
        ["Delta", "Gamma", "Theta", "Vega", "Rho"],
        index=0
    )
    
    # Gerar gráfico do Greek selecionado
    fig, ax = plt.subplots(figsize=(10, 5))
    S_range = np.linspace(50, 150, 100)
    
    # Calcular Greek para cada S no intervalo
    valores_greek = []
    for s in S_range:
        _, d, g, t, v, r = black_scholes_greeks(s, K, T, r, sigma, tipo_opcao)
        if greek_selecionado == "Delta":
            valores_greek.append(d)
        elif greek_selecionado == "Gamma":
            valores_greek.append(g)
        elif greek_selecionado == "Theta":
            valores_greek.append(t / 365)  # Theta diário
        elif greek_selecionado == "Vega":
            valores_greek.append(v)
        else:
            valores_greek.append(r)
    
    ax.plot(S_range, valores_greek, color='darkorange', linewidth=2)
    ax.axvline(S, color='red', linestyle='--', label='Preço Atual (S)')
    ax.set_title(f"{greek_selecionado} vs Preço do Ativo", fontweight='bold')
    ax.set_xlabel("Preço do Ativo (S)")
    ax.set_ylabel(f"{greek_selecionado}")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)