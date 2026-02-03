import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# Ejemplo de equity
# --------------------------
# Supongamos que tu equity diario es algo así:
dates = pd.date_range(start="2025-01-01", periods=250, freq='B')  # 250 días hábiles
np.random.seed(42)
returns = np.random.normal(0.0003, 0.01, size=len(dates))  # pequeños retornos diarios
equity = pd.Series(100000 * (1 + returns).cumprod(), index=dates)

# --------------------------
# Calcular CAGR
# --------------------------
years = (equity.index[-1] - equity.index[0]).days / 365.25
cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
print(f"CAGR anualizado: {cagr*100:.2f}%")

# --------------------------
# Crear curva CAGR teórica (crecimiento constante)
# --------------------------
equity_cagr = equity.iloc[0] * (1 + cagr) ** np.linspace(0, years, len(equity))

# --------------------------
# Plot
# --------------------------
plt.figure(figsize=(12,6))
plt.plot(equity.index, equity, label='Equity real')
plt.plot(equity.index, equity_cagr, label='Equity CAGR (crecimiento anual compuesto)', linestyle='--')
plt.title('Equity vs CAGR')
plt.xlabel('Fecha')
plt.ylabel('Capital')
plt.legend()
plt.show()
