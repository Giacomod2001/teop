import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configurazione pagina
st.set_page_config(
    page_title="Quantum Energy Trading Demo",
    page_icon="⚡",
    layout="wide"
)

# Titolo e descrizione
st.title("⚡ Quantum Energy Trading System")
st.markdown("""
### Simulazione di Trading Energetico con Ottimizzazione Quantistica
Sistema di trading per mercati energetici che utilizza algoritmi quantum-inspired per ottimizzare
acquisto/vendita di energia, bilanciamento della rete e gestione dello storage.
""")

# Sidebar per parametri
st.sidebar.header("⚙️ Parametri di Trading Energetico")

# Parametri di input
market_type = st.sidebar.selectbox(
    "Tipo di Mercato",
    ["Day-Ahead", "Intraday", "Balancing Market", "Ancillary Services"]
)

n_hours = st.sidebar.slider("Ore di Simulazione", 24, 720, 168)  # default 1 settimana
n_nodes = st.sidebar.slider("Nodi della Rete", 3, 10, 5)
storage_capacity = st.sidebar.slider("Capacità Storage (MWh)", 0, 1000, 500)
max_trading_volume = st.sidebar.slider("Volume Max Trading (MW)", 10, 500, 100)
quantum_layers = st.sidebar.slider("Strati Quantistici", 2, 8, 4)
risk_factor = st.sidebar.slider("Fattore di Rischio", 0.1, 1.0, 0.3)

# Parametri stagionali
season = st.sidebar.selectbox("Stagione", ["Estate", "Inverno", "Primavera", "Autunno"])
renewable_penetration = st.sidebar.slider("Penetrazione Rinnovabili (%)", 0, 100, 40)

# Classe per simulazione mercato energetico
class EnergyMarketSimulator:
    def __init__(self, n_hours, n_nodes, season, renewable_penetration):
        self.n_hours = n_hours
        self.n_nodes = n_nodes
        self.season = season
        self.renewable_penetration = renewable_penetration / 100
        
    def generate_base_load(self):
        """Genera profilo di carico base"""
        hours = np.arange(self.n_hours)
        daily_pattern = np.tile([
            0.7, 0.65, 0.6, 0.58, 0.6, 0.65,  # 00:00 - 05:00
            0.7, 0.8, 0.9, 0.95, 0.98, 1.0,    # 06:00 - 11:00
            0.98, 0.95, 0.93, 0.95, 0.98, 1.0,  # 12:00 - 17:00
            0.95, 0.9, 0.85, 0.8, 0.75, 0.72   # 18:00 - 23:00
        ], self.n_hours // 24 + 1)[:self.n_hours]
        
        # Variazione stagionale
        seasonal_factor = {
            "Estate": 1.2,
            "Inverno": 1.3,
            "Primavera": 1.0,
            "Autunno": 1.1
        }[self.season]
        
        base_load = daily_pattern * seasonal_factor * 1000  # in MW
        # Aggiungi rumore
        noise = np.random.normal(0, 50, self.n_hours)
        
        return base_load + noise
    
    def generate_renewable_generation(self):
        """Genera produzione rinnovabile (solare + eolico)"""
        hours = np.arange(self.n_hours)
        
        # Solare (picco a mezzogiorno)
        solar_pattern = np.tile([
            0, 0, 0, 0, 0, 0.1,           # 00:00 - 05:00
            0.3, 0.5, 0.7, 0.85, 0.95, 1.0,  # 06:00 - 11:00
            1.0, 0.95, 0.85, 0.7, 0.5, 0.3,  # 12:00 - 17:00
            0.1, 0, 0, 0, 0, 0             # 18:00 - 23:00
        ], self.n_hours // 24 + 1)[:self.n_hours]
        
        # Eolico (più casuale)
        wind_pattern = 0.3 + 0.4 * np.sin(hours * 0.1) + np.random.normal(0, 0.2, self.n_hours)
        wind_pattern = np.clip(wind_pattern, 0, 1)
        
        # Combina solare ed eolico
        renewable = (0.6 * solar_pattern + 0.4 * wind_pattern) * self.renewable_penetration * 800
        
        return renewable
    
    def generate_energy_prices(self, demand, supply):
        """Genera prezzi dell'energia basati su domanda/offerta"""
        # Prezzo base
        base_price = 50  # €/MWh
        
        # Fattore di scarsità
        scarcity = (demand - supply) / demand
        scarcity = np.clip(scarcity, -0.5, 2.0)
        
        # Calcola prezzi con volatilità
        prices = base_price * (1 + scarcity) * (1 + np.random.normal(0, 0.1, len(demand)))
        prices = np.maximum(prices, 0)  # No prezzi negativi (per semplicità)
        
        return prices

# Classe per Quantum Energy Trading
class QuantumEnergyTrader:
    def __init__(self, quantum_layers, risk_factor, storage_capacity, max_volume):
        self.quantum_layers = quantum_layers
        self.risk_factor = risk_factor
        self.storage_capacity = storage_capacity
        self.max_volume = max_volume
        self.storage_level = storage_capacity / 2  # Inizia a metà capacità
        
    def quantum_state_preparation(self, market_data):
        """Prepara stati quantistici dai dati di mercato"""
        n_qubits = self.quantum_layers
        n_states = 2 ** n_qubits
        
        # Genera stati possibili (strategie di trading)
        states = []
        for i in range(n_states):
            # Ogni stato rappresenta una strategia: [buy_threshold, sell_threshold, storage_strategy]
            state = {
                'buy_threshold': np.random.uniform(0.2, 0.8),
                'sell_threshold': np.random.uniform(0.5, 1.0),
                'storage_rate': np.random.uniform(0.1, 0.5),
                'amplitude': np.random.random()
            }
            states.append(state)
        
        # Normalizza ampiezze
        total_amp = sum(s['amplitude'] for s in states)
        for s in states:
            s['amplitude'] /= total_amp
            
        return states
    
    def quantum_interference(self, states, market_conditions):
        """Simula interferenza quantistica tra stati"""
        # Calcola fitness per ogni stato basato sulle condizioni di mercato
        for state in states:
            # Valuta la strategia
            profit_potential = self.evaluate_strategy(state, market_conditions)
            risk_penalty = state['buy_threshold'] * self.risk_factor
            
            # Interferenza costruttiva/distruttiva
            state['fitness'] = profit_potential - risk_penalty
            state['amplitude'] *= np.exp(state['fitness'])
        
        # Rinormalizza
        total_amp = sum(s['amplitude'] for s in states)
        for s in states:
            s['amplitude'] /= total_amp
            
        return states
    
    def evaluate_strategy(self, state, market_conditions):
        """Valuta una strategia di trading"""
        price_volatility = market_conditions['volatility']
        price_trend = market_conditions['trend']
        
        # Stima profitto potenziale
        spread = state['sell_threshold'] - state['buy_threshold']
        volume_factor = 1 - abs(0.5 - state['storage_rate'])
        
        profit = spread * volume_factor * (1 + price_trend) / (1 + price_volatility)
        
        return profit
    
    def quantum_measurement(self, states):
        """Collasso della funzione d'onda - selezione strategia"""
        # Probabilità basate sulle ampiezze
        probabilities = [s['amplitude'] ** 2 for s in states]
        probabilities = np.array(probabilities) / sum(probabilities)
        
        # Seleziona stato
        selected_idx = np.random.choice(len(states), p=probabilities)
        
        return states[selected_idx]
    
    def generate_trading_signals(self, prices, demand, supply):
        """Genera segnali di trading usando ottimizzazione quantistica"""
        signals = []
        positions = []
        storage_levels = []
        
        for i in range(len(prices)):
            # Prepara condizioni di mercato
            market_conditions = {
                'price': prices[i],
                'demand': demand[i],
                'supply': supply[i],
                'volatility': np.std(prices[max(0, i-24):i+1]) if i > 0 else 10,
                'trend': np.mean(np.diff(prices[max(0, i-24):i])) if i > 1 else 0
            }
            
            # Prepara stati quantistici
            states = self.quantum_state_preparation(market_conditions)
            
            # Applica interferenza quantistica
            states = self.quantum_interference(states, market_conditions)
            
            # Misura (seleziona strategia)
            strategy = self.quantum_measurement(states)
            
            # Genera segnale basato sulla strategia
            price_normalized = (prices[i] - np.min(prices)) / (np.max(prices) - np.min(prices) + 0.001)
            
            if price_normalized < strategy['buy_threshold'] and self.storage_level < self.storage_capacity:
                # Compra energia
                volume = min(self.max_volume, self.storage_capacity - self.storage_level)
                signal = 1
                position = volume
                self.storage_level += volume * strategy['storage_rate']
            elif price_normalized > strategy['sell_threshold'] and self.storage_level > 0:
                # Vendi energia
                volume = min(self.max_volume, self.storage_level)
                signal = -1
                position = -volume
                self.storage_level -= volume * strategy['storage_rate']
            else:
                # Hold
                signal = 0
                position = 0
            
            signals.append(signal)
            positions.append(position)
            storage_levels.append(self.storage_level)
            
            # Limita storage level
            self.storage_level = np.clip(self.storage_level, 0, self.storage_capacity)
        
        return signals, positions, storage_levels

# Genera dati di mercato
@st.cache_data
def generate_market_data(n_hours, n_nodes, season, renewable_penetration):
    simulator = EnergyMarketSimulator(n_hours, n_nodes, season, renewable_penetration)
    
    # Genera componenti
    demand = simulator.generate_base_load()
    renewable = simulator.generate_renewable_generation()
    conventional = demand - renewable + np.random.normal(0, 20, n_hours)
    supply = renewable + conventional
    
    # Genera prezzi
    prices = simulator.generate_energy_prices(demand, supply)
    
    # Crea DataFrame
    time_index = pd.date_range(start=datetime.now(), periods=n_hours, freq='H')
    
    df = pd.DataFrame({
        'demand': demand,
        'supply': supply,
        'renewable': renewable,
        'conventional': conventional,
        'price': prices
    }, index=time_index)
    
    return df

# Genera dati
market_data = generate_market_data(n_hours, n_nodes, season, renewable_penetration)

# Inizializza trader quantistico
quantum_trader = QuantumEnergyTrader(quantum_layers, risk_factor, storage_capacity, max_trading_volume)

# Genera segnali di trading
with st.spinner('Esecuzione ottimizzazione quantistica...'):
    signals, positions, storage_levels = quantum_trader.generate_trading_signals(
        market_data['price'].values,
        market_data['demand'].values,
        market_data['supply'].values
    )

# Calcola profitti
trades_df = pd.DataFrame({
    'signal': signals,
    'position': positions,
    'storage': storage_levels,
    'price': market_data['price'].values
}, index=market_data.index)

# Calcola P&L
trades_df['revenue'] = -trades_df['position'] * trades_df['price']  # Negativo quando compra, positivo quando vende
trades_df['cumulative_revenue'] = trades_df['revenue'].cumsum()

# Metriche principali
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = trades_df['revenue'].sum()
    st.metric("Ricavo Totale", f"€{total_revenue:,.0f}")

with col2:
    n_trades = (trades_df['signal'] != 0).sum()
    st.metric("Numero Trade", f"{n_trades}")

with col3:
    avg_price = market_data['price'].mean()
    st.metric("Prezzo Medio", f"€{avg_price:.2f}/MWh")

with col4:
    price_volatility = market_data['price'].std()
    st.metric("Volatilità Prezzo", f"€{price_volatility:.2f}")

# Grafici principali
st.subheader("📊 Dinamiche del Mercato Energetico")

# Grafico domanda/offerta e prezzi
fig1 = make_subplots(
    rows=3, cols=1,
    subplot_titles=('Domanda vs Offerta', 'Prezzi Energia', 'Storage e Trading'),
    vertical_spacing=0.1,
    row_heights=[0.35, 0.35, 0.3]
)

# Domanda e Offerta
fig1.add_trace(
    go.Scatter(x=market_data.index, y=market_data['demand'],
               name='Domanda', line=dict(color='red', width=2)),
    row=1, col=1
)
fig1.add_trace(
    go.Scatter(x=market_data.index, y=market_data['supply'],
               name='Offerta Totale', line=dict(color='blue', width=2)),
    row=1, col=1
)
fig1.add_trace(
    go.Scatter(x=market_data.index, y=market_data['renewable'],
               name='Rinnovabili', line=dict(color='green', width=1, dash='dot')),
    row=1, col=1
)

# Prezzi con segnali di trading
fig1.add_trace(
    go.Scatter(x=market_data.index, y=market_data['price'],
               name='Prezzo', line=dict(color='purple', width=2)),
    row=2, col=1
)

# Aggiungi marcatori per buy/sell
buy_signals = trades_df[trades_df['signal'] == 1]
sell_signals = trades_df[trades_df['signal'] == -1]

fig1.add_trace(
    go.Scatter(x=buy_signals.index, y=buy_signals['price'],
               mode='markers', name='Buy',
               marker=dict(color='green', size=10, symbol='triangle-up')),
    row=2, col=1
)
fig1.add_trace(
    go.Scatter(x=sell_signals.index, y=sell_signals['price'],
               mode='markers', name='Sell',
               marker=dict(color='red', size=10, symbol='triangle-down')),
    row=2, col=1
)

# Storage level
fig1.add_trace(
    go.Scatter(x=trades_df.index, y=trades_df['storage'],
               name='Livello Storage', fill='tozeroy',
               line=dict(color='orange', width=2)),
    row=3, col=1
)

fig1.update_xaxes(title_text="Tempo", row=3, col=1)
fig1.update_yaxes(title_text="MW", row=1, col=1)
fig1.update_yaxes(title_text="€/MWh", row=2, col=1)
fig1.update_yaxes(title_text="MWh", row=3, col=1)

fig1.update_layout(height=800, showlegend=True)
st.plotly_chart(fig1, use_container_width=True)

# Analisi Quantistica
st.subheader("⚛️ Analisi Quantistica del Trading")

col1, col2 = st.columns(2)

with col1:
    # Distribuzione dei prezzi di acquisto/vendita
    fig2 = go.Figure()
    
    if len(buy_signals) > 0:
        fig2.add_trace(go.Histogram(x=buy_signals['price'], name='Prezzi Acquisto',
                                    marker_color='green', opacity=0.7))
    if len(sell_signals) > 0:
        fig2.add_trace(go.Histogram(x=sell_signals['price'], name='Prezzi Vendita',
                                    marker_color='red', opacity=0.7))
    
    fig2.update_layout(title="Distribuzione Prezzi Trading",
                      xaxis_title="Prezzo (€/MWh)",
                      yaxis_title="Frequenza",
                      height=400)
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    # Performance cumulativa
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=trades_df.index, y=trades_df['cumulative_revenue'],
                              mode='lines', name='Ricavo Cumulativo',
                              line=dict(color='cyan', width=2)))
    
    fig3.update_layout(title="Performance Cumulativa",
                      xaxis_title="Tempo",
                      yaxis_title="Ricavo (€)",
                      height=400)
    st.plotly_chart(fig3, use_container_width=True)

# Heatmap oraria
st.subheader("🗓️ Pattern Orari di Trading")

# Prepara dati per heatmap
trades_df['hour'] = trades_df.index.hour
trades_df['day'] = trades_df.index.day_name()

# Crea matrice per heatmap
pivot_revenue = trades_df.pivot_table(values='revenue', index='hour', columns='day', aggfunc='mean')

fig4 = go.Figure(data=go.Heatmap(
    z=pivot_revenue.values,
    x=pivot_revenue.columns,
    y=pivot_revenue.index,
    colorscale='RdYlGn',
    zmid=0,
    text=np.round(pivot_revenue.values, 1),
    texttemplate='%{text}',
    textfont={"size": 10}
))

fig4.update_layout(
    title="Ricavo Medio per Ora del Giorno",
    xaxis_title="Giorno",
    yaxis_title="Ora",
    height=500
)

st.plotly_chart(fig4, use_container_width=True)

# Statistiche di trading
st.subheader("📈 Statistiche di Trading")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Operazioni di Acquisto**")
    buy_stats = trades_df[trades_df['signal'] == 1]['price'].describe()
    st.dataframe(buy_stats.to_frame().style.format("{:.2f}"))

with col2:
    st.markdown("**Operazioni di Vendita**")
    sell_stats = trades_df[trades_df['signal'] == -1]['price'].describe()
    st.dataframe(sell_stats.to_frame().style.format("{:.2f}"))

with col3:
    st.markdown("**Efficienza Storage**")
    storage_efficiency = {
        'Utilizzo Medio (%)': (trades_df['storage'].mean() / storage_capacity * 100),
        'Utilizzo Max (%)': (trades_df['storage'].max() / storage_capacity * 100),
        'Turnover': n_trades / (n_hours / 24),
        'Spread Medio (€)': (sell_signals['price'].mean() - buy_signals['price'].mean()) if len(buy_signals) > 0 and len(sell_signals) > 0 else 0
    }
    st.dataframe(pd.DataFrame(storage_efficiency, index=[0]).T.style.format("{:.2f}"))

# Info box
st.info("""
**Sistema Quantum Energy Trading:**
- **Stati Quantistici**: Ogni stato rappresenta una strategia di trading con parametri specifici
- **Interferenza**: Le strategie interferiscono basandosi sulle condizioni di mercato
- **Misurazione**: Il sistema "collassa" sulla strategia ottimale per ogni ora
- **Storage Management**: Ottimizzazione quantistica del livello di storage energetico
- **Risk Management**: Il fattore di rischio influenza la selezione delle strategie
""")

# Warning
st.warning("""
⚠️ **Disclaimer**: Questa è una simulazione dimostrativa per scopi educativi. 
I mercati energetici reali sono molto più complessi e richiedono considerazioni aggiuntive come:
- Vincoli di rete e congestioni
- Requisiti di bilanciamento in tempo reale
- Contratti bilaterali e mercati forward
- Regolamentazione e vincoli operativi
""")
