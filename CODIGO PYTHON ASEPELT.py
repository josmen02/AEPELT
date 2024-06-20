# Librerías
# Tratamiento de datos
#=====================================================================
import numpy as np
import pandas as pd
from skforecast.datasets import fetch_dataset
from sas7bdat import SAS7BDAT

# Gráficos
#=====================================================================
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 10

# Modelado y Forecasting
#=====================================================================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster

# Configuración warnings
#=====================================================================
import warnings
warnings.filterwarnings('once')
In [ ]:
with SAS7BDAT('datos/cargas.sas7bdat') as archivo:
# Lee el archivo SAS y guarda los datos en un DataFrame de pandas
datos = archivo.to_data_frame()
In [ ]:
datos = pd.read_excel('cierres_empresas_preparado.xlsx')
In [ ]:
# Preparación del dato
#=====================================================================
datos['fecha'] = pd.to_datetime(datos['Fecha'], format='%Y-%m-%d')
datos = datos.set_index('fecha')
datos = datos.asfreq('D')
datos = datos.sort_index()
datos.head()
In [ ]:
cierres = datos.ffill()
cierres['Fecha'] = datos.index
cierres.head()
In [ ]:
#cierres.to_excel('cierres_imputado.xlsx', index=False)
In [ ]:
with SAS7BDAT('datos/factores.sas7bdat') as archivo:
# Lee el archivo SAS y guarda los datos en un DataFrame de pandas
factores = archivo.to_data_frame()
In [ ]:
# Preparación del dato
#=====================================================================
factores['Fecha'] = pd.to_datetime(factores['Fecha'], format='%Y-%m-%d')
factores = factores.set_index('Fecha')
factores = factores.asfreq('D')
factores = factores.sort_index()
factores.head()
In [ ]:
# Separación datos train-test
#=====================================================================
steps = 14
factores_train = factores[:-steps]
factores_test  = factores[-steps:]
factores_test2  = factores[-steps-1:]

factores_train.to_excel('factores_train.xlsx', index=False)
factores_test.to_excel('factores_test.xlsx', index=False)
In [ ]:
print(f"Fechas train : {factores_train.index.min()} --- {factores_train.index.max()}  (n={len(factores_train)})")
print(f"Fechas test  : {factores_test.index.min()} --- {factores_test.index.max()}  (n={len(factores_test)})")

fig, ax = plt.subplots(figsize=(6, 2.5))
factores_train['Bancario_y_grandes_empresas'].plot(ax=ax, label='train')
factores_test['Bancario_y_grandes_empresas'].plot(ax=ax, label='test')
ax.legend();
In [ ]:
print(f"Fechas train : {factores_train.index.min()} --- {factores_train.index.max()}  (n={len(factores_train)})")
print(f"Fechas test  : {factores_test.index.min()} --- {factores_test.index.max()}  (n={len(factores_test)})")

fig, ax = plt.subplots(figsize=(6, 2.5))
factores_train['Transporte_y_tecnologia'].plot(ax=ax, label='train')
factores_test['Transporte_y_tecnologia'].plot(ax=ax, label='test')
ax.legend();
In [ ]:
print(f"Fechas train : {factores_train.index.min()} --- {factores_train.index.max()}  (n={len(factores_train)})")
print(f"Fechas test  : {factores_test.index.min()} --- {factores_test.index.max()}  (n={len(factores_test)})")

fig, ax = plt.subplots(figsize=(6, 2.5))
factores_train['Industrial'].plot(ax=ax, label='train')
factores_test['Industrial'].plot(ax=ax, label='test')
ax.legend();
In [ ]:
print(f"Fechas train : {factores_train.index.min()} --- {factores_train.index.max()}  (n={len(factores_train)})")
print(f"Fechas test  : {factores_test.index.min()} --- {factores_test.index.max()}  (n={len(factores_test)})")

fig, ax = plt.subplots(figsize=(6, 2.5))
factores_train['Acciona'].plot(ax=ax, label='train')
factores_test['Acciona'].plot(ax=ax, label='test')
ax.legend();
In [ ]:
print(f"Fechas train : {factores_train.index.min()} --- {factores_train.index.max()}  (n={len(factores_train)})")
print(f"Fechas test  : {factores_test.index.min()} --- {factores_test.index.max()}  (n={len(factores_test)})")

fig, ax = plt.subplots(figsize=(6, 2.5))
factores_train['Energetico'].plot(ax=ax, label='train')
factores_test['Energetico'].plot(ax=ax, label='test')
ax.legend();
#ARMA - Garch
In [ ]:
import pmdarima
from statsmodels.tsa.arima.model import ARIMA
import arch
In [ ]:
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
from arch import arch_model
In [ ]:
from scipy.optimize import minimize

#Bancario_y_grandes_empresas
In [ ]:
Bancario_y_grandes_empresas = factores_train.iloc[1110:]['Bancario_y_grandes_empresas']
In [ ]:
model = ARIMA(Bancario_y_grandes_empresas, order=(0, 1, 2))
arima_model = model.fit()

arima_residuals = arima_model.resid*10
In [ ]:
# fit a GARCH(1,1) model on the residuals of the ARIMA model
garch = arch.arch_model(arima_residuals, p=1, q=1)
garch_fit = garch.fit()
In [ ]:
def loss_function(params, *args):
data = args[0]
p = int(params[0])
q = int(params[1])
model = arch_model(data, vol='GARCH', p=p, q=q)
result = model.fit(disp='off')
return result.aic

# Datos de ejemplo
data = pd.Series(np.random.normal(size=100))

# Optimización de los parámetros
initial_guess = [1, 1]  # Valores iniciales para p y q
bounds = [(0, None), (0, None)]  # Límites para los parámetros
result = minimize(loss_function, initial_guess, args=(Bancario_y_grandes_empresas,), bounds=bounds)

# Parámetros óptimos encontrados
optimal_params = result.x
print("Parámetros óptimos:", optimal_params)
In [ ]:
garch_fit.summary()
In [ ]:
# Hacer predicciones fuera de muestra con el modelo GARCH
forecast_horizon = 14
forecast_arima = arima_model.forecast(steps=14)
forecast_resid = (garch_fit.forecast(horizon=forecast_horizon)).variance.iloc[-1]/10
forecast_resid.index = forecast_arima.index
forecast_garch = forecast_arima + forecast_resid
In [ ]:
# Genera las fechas para los próximos 14 días
last_date = Bancario_y_grandes_empresas.index[-1]
forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1)

# Visualiza los resultados
plt.figure(figsize=(10, 6))
plt.plot(Bancario_y_grandes_empresas.index, Bancario_y_grandes_empresas, label='Actual Returns', color='blue')
plt.plot(forecast_dates[1:], forecast_garch, label='Forecasted Returns', color='red')
plt.plot(factores_test2['Bancario_y_grandes_empresas'].index,factores_test2['Bancario_y_grandes_empresas'], label = 'test', color='green')
plt.xlabel('Date')
plt.ylabel('Bancario_y_grandes_empresas')
plt.title('GARCH Forecast for Next 14 Days')
plt.legend()
plt.grid(True)
plt.show()
In [ ]:
#pred_garch=pd.DataFrame()
In [ ]:
pred_garch['FORECAST_FACT1']=forecast_garch
In [ ]:
pred_garch.to_excel('pred_garch.xlsx', index=False)

#Transporte_y_tecnologia
In [ ]:
Transporte_y_tecnologia = factores_train.iloc[1110:]['Transporte_y_tecnologia']
In [ ]:
model = ARIMA(Transporte_y_tecnologia, order=(0, 1, 0))
arima_model = model.fit()

arima_residuals = arima_model.resid*10
In [ ]:
def loss_function(params, *args):
data = args[0]
p = int(params[0])
q = int(params[1])
model = arch_model(data, vol='GARCH', p=p, q=q)
result = model.fit(disp='off')
return result.aic

# Optimización de los parámetros
initial_guess = [1, 1]  # Valores iniciales para p y q
bounds = [(0, None), (0, None)]  # Límites para los parámetros
result = minimize(loss_function, initial_guess, args=(arima_residuals,), bounds=bounds)

# Parámetros óptimos encontrados
optimal_params = result.x
print("Parámetros óptimos:", optimal_params)
In [ ]:
# fit a GARCH(1,1) model on the residuals of the ARIMA model
garch = arch.arch_model(arima_residuals,p=1,  q=1)
garch_fit = garch.fit()
In [ ]:
garch_fit.summary()
In [ ]:
# Hacer predicciones fuera de muestra con el modelo GARCH
forecast_horizon = 14
forecast_arima = arima_model.forecast(steps=14)
forecast_resid = (garch_fit.forecast(horizon=forecast_horizon)).variance.iloc[-1]/10
forecast_resid.index = forecast_arima.index
forecast_garch = forecast_arima + forecast_resid
In [ ]:
# Genera las fechas para los próximos 14 días
last_date = Transporte_y_tecnologia.index[-1]
forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1)

# Visualiza los resultados
plt.figure(figsize=(10, 6))
plt.plot(Transporte_y_tecnologia.index, Transporte_y_tecnologia, label='Actual Returns', color='blue')
plt.plot(forecast_dates[1:], forecast_garch, label='Forecasted Returns', color='red')
plt.plot(factores_test2['Transporte_y_tecnologia'].index,factores_test2['Transporte_y_tecnologia'], label = 'test', color='green')
plt.xlabel('Date')
plt.ylabel('Transporte_y_tecnologia')
plt.title('GARCH Forecast for Next 14 Days')
plt.legend()
plt.grid(True)
plt.show()
In [ ]:
forecast_garch
In [ ]:
pred_garch['FORECAST_FACT2']=forecast_garch
In [ ]:
pred_garch.to_excel('pred_garch.xlsx', index=False)

#Industrial
In [ ]:
Industrial = factores_train.iloc[1110:]['Industrial']
In [ ]:
model = ARIMA(Industrial, order=(0, 1, 0))
arima_model = model.fit()

arima_residuals = arima_model.resid*10
In [ ]:
def loss_function(params, *args):
data = args[0]
p = int(params[0])
q = int(params[1])
model = arch_model(data, vol='GARCH', p=p, q=q)
result = model.fit(disp='off')
return result.aic

# Optimización de los parámetros
initial_guess = [1, 1]  # Valores iniciales para p y q
bounds = [(0, None), (0, None)]  # Límites para los parámetros
result = minimize(loss_function, initial_guess, args=(arima_residuals,), bounds=bounds)

# Parámetros óptimos encontrados
optimal_params = result.x
print("Parámetros óptimos:", optimal_params)
In [ ]:
# fit a GARCH(1,1) model on the residuals of the ARIMA model
garch = arch.arch_model(arima_residuals,p=1,  q=1)
garch_fit = garch.fit()
In [ ]:
garch_fit.summary()
In [ ]:
# Hacer predicciones fuera de muestra con el modelo GARCH
forecast_horizon = 14
forecast_arima = arima_model.forecast(steps=14)
forecast_resid = ((garch_fit.forecast(horizon=forecast_horizon)).variance.iloc[-1])/10
forecast_resid.index = forecast_arima.index
forecast_garch = forecast_arima + forecast_resid
In [ ]:
forecast_resid
In [ ]:
# Genera las fechas para los próximos 14 días
last_date = Industrial.index[-1]
forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1)

# Visualiza los resultados
plt.figure(figsize=(10, 6))
plt.plot(Industrial.index, Industrial, label='Actual Returns', color='blue')
plt.plot(forecast_dates[1:], forecast_garch, label='Forecasted Returns', color='red')
plt.plot(factores_test2['Industrial'].index,factores_test2['Industrial'], label = 'test', color='green')
plt.xlabel('Date')
plt.ylabel('Transporte_y_tecnologia')
plt.title('GARCH Forecast for Next 14 Days')
plt.legend()
plt.grid(True)
plt.show()
In [ ]:
pred_garch['FORECAST_FACT3']=forecast_garch
In [ ]:
pred_garch.to_excel('pred_garch.xlsx', index=False)

#Acciona
In [ ]:
Acciona = factores_train.iloc[1110:]['Acciona']
In [ ]:
model = ARIMA(Acciona, order=(0, 1, 0))
arima_model = model.fit()

arima_residuals = arima_model.resid*10
In [ ]:
# fit a GARCH(1,1) model on the residuals of the ARIMA model
garch = arch.arch_model(arima_residuals,p=1,  q=1)
garch_fit = garch.fit()
In [ ]:
garch_fit.summary()
In [ ]:
# Hacer predicciones fuera de muestra con el modelo GARCH
forecast_horizon = 14
forecast_arima = arima_model.forecast(steps=14)
forecast_resid = ((garch_fit.forecast(horizon=forecast_horizon)).variance.iloc[-1])/10
forecast_resid.index = forecast_arima.index
forecast_garch = forecast_arima + forecast_resid
In [ ]:
# Genera las fechas para los próximos 14 días
last_date = Acciona.index[-1]
forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1)

# Visualiza los resultados
plt.figure(figsize=(10, 6))
plt.plot(Acciona.index, Acciona, label='Actual Returns', color='blue')
plt.plot(forecast_dates[1:], forecast_garch, label='Forecasted Returns', color='red')
plt.plot(factores_test2['Acciona'].index,factores_test2['Acciona'], label = 'test', color='green')
plt.xlabel('Date')
plt.ylabel('Transporte_y_tecnologia')
plt.title('GARCH Forecast for Next 14 Days')
plt.legend()
plt.grid(True)
plt.show()
In [ ]:
pred_garch['FORECAST_FACT4']=forecast_garch
In [ ]:
pred_garch.to_excel('pred_garch.xlsx', index=False)

#Energetico
In [ ]:
Energetico = factores_train.iloc[1110:]['Energetico']
In [ ]:
model = ARIMA(Energetico, order=(0, 1, 0))
arima_model = model.fit()

arima_residuals = arima_model.resid*10
In [ ]:
def loss_function(params, *args):
data = args[0]
p = int(params[0])
q = int(params[1])
model = arch_model(data, vol='GARCH', p=p, q=q)
result = model.fit(disp='off')
return result.aic

# Optimización de los parámetros
initial_guess = [1, 1]  # Valores iniciales para p y q
bounds = [(0, None), (0, None)]  # Límites para los parámetros
result = minimize(loss_function, initial_guess, args=(arima_residuals,), bounds=bounds)

# Parámetros óptimos encontrados
optimal_params = result.x
print("Parámetros óptimos:", optimal_params)
In [ ]:
# fit a GARCH(1,1) model on the residuals of the ARIMA model
garch = arch.arch_model(arima_residuals,p=1,  q=1)
garch_fit = garch.fit()
In [ ]:
garch_fit.summary()
In [ ]:
# Hacer predicciones fuera de muestra con el modelo GARCH
forecast_horizon = 14
forecast_arima = arima_model.forecast(steps=14)
forecast_resid = ((garch_fit.forecast(horizon=forecast_horizon)).variance.iloc[-1])/10
forecast_resid.index = forecast_arima.index
forecast_garch = forecast_arima + forecast_resid
In [ ]:
# Genera las fechas para los próximos 14 días
last_date = Energetico.index[-1]
forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1)

# Visualiza los resultados
plt.figure(figsize=(10, 6))
plt.plot(Energetico.index, Energetico, label='Actual Returns', color='blue')
plt.plot(forecast_dates[1:], forecast_garch, label='Forecasted Returns', color='red')
plt.plot(factores_test2['Energetico'].index,factores_test2['Energetico'], label = 'test', color='green')
plt.xlabel('Date')
plt.ylabel('Transporte_y_tecnologia')
plt.title('GARCH Forecast for Next 14 Days')
plt.legend()
plt.grid(True)
plt.show()
In [ ]:
pred_garch['FORECAST_FACT5']=forecast_garch
In [ ]:
pred_garch.to_excel('pred_garch.xlsx', index=False)

#Random Forest Autorregresivo
In [ ]:
#pred_rf=pd.DataFrame()
#pred_rf.to_excel('pred_rf.xlsx', index=False)
#Bancario_y_grandes_empresas
In [ ]:
# Crear y entrenar forecaster
#=====================================================================
forecaster = ForecasterAutoreg(
regressor = RandomForestRegressor(random_state=123),
lags = 14
)

forecaster.fit(y=factores_train['Bancario_y_grandes_empresas'])
forecaster.summary()

# Predicciones
#=====================================================================
steps = 14
predicciones = forecaster.predict(steps=steps)
predicciones.head(14)

# Gráfico de predicciones vs valores reales
#=====================================================================
fig, ax = plt.subplots(figsize=(6, 2.5))
factores_test['Bancario_y_grandes_empresas'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();
In [ ]:
pred_rf['FORECAST_FACT1']=predicciones
In [ ]:
# Calcular métricas de ajuste
y_true = factores_test['Bancario_y_grandes_empresas']
mse = (mean_squared_error(y_true, predicciones))
rmse = np.sqrt(mean_squared_error(y_true, predicciones))
mae = mean_absolute_error(y_true, predicciones)
r2 = r2_score(y_true, predicciones)

# Mostrar resultados de métricas
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

#Transporte_y_tecnologia
In [ ]:
# Crear y entrenar forecaster
#=====================================================================
forecaster = ForecasterAutoreg(
regressor = RandomForestRegressor(random_state=123),
lags = 14
)

forecaster.fit(y=factores_train['Transporte_y_tecnologia'])
forecaster.summary()

# Predicciones
#=====================================================================
steps = 14
predicciones = forecaster.predict(steps=steps)
predicciones.head(14)

# Gráfico de predicciones vs valores reales
#=====================================================================
fig, ax = plt.subplots(figsize=(6, 2.5))
factores_test['Transporte_y_tecnologia'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();
In [ ]:
# Calcular métricas de ajuste
y_true = factores_test['Transporte_y_tecnologia']
mse = (mean_squared_error(y_true, predicciones))
rmse = np.sqrt(mean_squared_error(y_true, predicciones))
mae = mean_absolute_error(y_true, predicciones)
r2 = r2_score(y_true, predicciones)

# Mostrar resultados de métricas
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
In [ ]:
pred_rf['FORECAST_FACT2']=predicciones

#Industrial
In [ ]:
# Crear y entrenar forecaster
#=====================================================================
forecaster = ForecasterAutoreg(
regressor = RandomForestRegressor(random_state=123),
lags = 14
)

forecaster.fit(y=factores_train['Industrial'])
forecaster.summary()

# Predicciones
#=====================================================================
steps = 14
predicciones = forecaster.predict(steps=steps)
predicciones.head(14)

# Gráfico de predicciones vs valores reales
#=====================================================================
fig, ax = plt.subplots(figsize=(6, 2.5))
factores_test['Industrial'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();
In [ ]:
pred_rf['FORECAST_FACT3']=predicciones
In [ ]:
# Calcular métricas de ajuste
y_true = factores_test['Industrial']
mse = (mean_squared_error(y_true, predicciones))
rmse = np.sqrt(mean_squared_error(y_true, predicciones))
mae = mean_absolute_error(y_true, predicciones)
r2 = r2_score(y_true, predicciones)

# Mostrar resultados de métricas
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

#Acciona
In [ ]:
# Crear y entrenar forecaster
# ======================================================================
forecaster = ForecasterAutoreg(
regressor = RandomForestRegressor(random_state=123),
lags = 14
)

forecaster.fit(y=factores_train['Acciona'])
forecaster.summary()

# Predicciones
# ======================================================================
steps = 14
predicciones = forecaster.predict(steps=steps)
predicciones.head(14)

# Gráfico de predicciones vs valores reales
# ======================================================================
fig, ax = plt.subplots(figsize=(6, 2.5))
factores_test['Acciona'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();
In [ ]:
pred_rf['FORECAST_FACT4']=predicciones
In [ ]:
# Calcular métricas de ajuste
y_true = factores_test['Acciona']
mse = (mean_squared_error(y_true, predicciones))
rmse = np.sqrt(mean_squared_error(y_true, predicciones))
mae = mean_absolute_error(y_true, predicciones)
r2 = r2_score(y_true, predicciones)

# Mostrar resultados de métricas
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

#Energetico
In [ ]:
# Crear y entrenar forecaster
#=====================================================================
forecaster = ForecasterAutoreg(
regressor = RandomForestRegressor(random_state=123),
lags = 14,
)

forecaster.fit(y=factores_train['Energetico'])
forecaster.summary()

# Predicciones
#=====================================================================
steps = 14
predicciones = forecaster.predict(steps=steps)
predicciones.head(14)

# Gráfico de predicciones vs valores reales
#=====================================================================
fig, ax = plt.subplots(figsize=(6, 2.5))
factores_test['Energetico'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();
In [ ]:
pred_rf['FORECAST_FACT5']=predicciones
In [ ]:
# Calcular métricas de ajuste
y_true = factores_test['Energetico']
mse = (mean_squared_error(y_true, predicciones))
rmse = np.sqrt(mean_squared_error(y_true, predicciones))
mae = mean_absolute_error(y_true, predicciones)
r2 = r2_score(y_true, predicciones)

# Mostrar resultados de métricas
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
In [ ]:
#pred_rf.to_excel('pred_rf.xlsx', index=False)
# COMPARACIÓN DE MODELOS
#=====================================================================
with SAS7BDAT('datos/error_arima.sas7bdat') as archivo:
# Lee el archivo SAS y guarda los datos en un DataFrame de pandas
error_arima = archivo.to_data_frame()
In [ ]:
pred_garch = pd.read_excel('pred_garch.xlsx')
In [ ]:
pred_rf = pd.read_excel('pred_rf.xlsx')
In [ ]:
RMSE=pd.DataFrame()
In [ ]:
rmse_values = {}
rmse_values['Arima'] = {}
rmse_values['Esm'] = {}
rmse_values['Garch'] = {}
rmse_values['RF_Autoreg'] = {}
In [ ]:
rmse_values['Arima']['Bancario_y_grandes_empresas'] = np.sqrt(mean_squared_error(error_arima['Bancario_y_grandes_empresas'], error_arima['FORECAST_FACT1']))
rmse_values['Arima']['Transporte_y_tecnologia'] = np.sqrt(mean_squared_error(error_arima['Transporte_y_tecnologia'], error_arima['FORECAST_FACT2']))
rmse_values['Arima']['Industrial'] = np.sqrt(mean_squared_error(error_arima['Industrial'], error_arima['FORECAST_FACT3']))
rmse_values['Arima']['Acciona'] = np.sqrt(mean_squared_error(error_arima['Acciona'], error_arima['FORECAST_FACT4']))
rmse_values['Arima']['Energetico'] = np.sqrt(mean_squared_error(error_arima['Energetico'], error_arima['FORECAST_FACT5']))
In [ ]:
rmse_values['Esm']['Bancario_y_grandes_empresas'] = 0.04495687
rmse_values['Esm']['Transporte_y_tecnologia'] = 0.04137976
rmse_values['Esm']['Industrial'] = 0.09277408
rmse_values['Esm']['Acciona'] = 0.09424696
rmse_values['Esm']['Energetico'] = 0.43123286
In [ ]:
rmse_values['Garch']['Bancario_y_grandes_empresas'] = np.sqrt(mean_squared_error(error_arima['Bancario_y_grandes_empresas'], pred_garch['FORECAST_FACT1']))
rmse_values['Garch']['Transporte_y_tecnologia'] = np.sqrt(mean_squared_error(error_arima['Transporte_y_tecnologia'], pred_garch['FORECAST_FACT2']))
rmse_values['Garch']['Industrial'] = np.sqrt(mean_squared_error(error_arima['Industrial'], pred_garch['FORECAST_FACT3']))
rmse_values['Garch']['Acciona'] = np.sqrt(mean_squared_error(error_arima['Acciona'], error_arima['FORECAST_FACT4']))
rmse_values['Garch']['Energetico'] = np.sqrt(mean_squared_error(error_arima['Energetico'], error_arima['FORECAST_FACT5']))
In [ ]:
rmse_values['RF_Autoreg']['Bancario_y_grandes_empresas'] = np.sqrt(mean_squared_error(error_arima['Bancario_y_grandes_empresas'], pred_rf['FORECAST_FACT1']))
rmse_values['RF_Autoreg']['Transporte_y_tecnologia'] = np.sqrt(mean_squared_error(error_arima['Transporte_y_tecnologia'], pred_rf['FORECAST_FACT2']))
rmse_values['RF_Autoreg']['Industrial'] = np.sqrt(mean_squared_error(error_arima['Industrial'], pred_rf['FORECAST_FACT3']))
rmse_values['RF_Autoreg']['Acciona'] = np.sqrt(mean_squared_error(error_arima['Acciona'], pred_rf['FORECAST_FACT4']))
rmse_values['RF_Autoreg']['Energetico'] = np.sqrt(mean_squared_error(error_arima['Energetico'], pred_rf['FORECAST_FACT5']))

# GRÁFICO COMPARACIÓN DE MODELOS
#=====================================================================
In [ ]:
# Extraer los nombres de las variables dependientes
variables_dependientes = list(rmse_values['Arima'].keys())

# Crear una lista de índices para el eje x
indices = range(len(variables_dependientes))

# Extraer los valores de RMSE para ARIMA y ESM
rmse_arima = [rmse_values['Arima'][variable] for variable in variables_dependientes]
rmse_esm = [rmse_values['Esm'][variable] for variable in variables_dependientes]
rmse_garch = [rmse_values['Garch'][variable] for variable in variables_dependientes]
rmse_rf = [rmse_values['RF_Autoreg'][variable] for variable in variables_dependientes]
# Crear el gráfico
plt.figure(figsize=(10, 6))

# Graficar las líneas para ARIMA y ESM
plt.plot(indices, rmse_arima, marker='o', label='ARIMA', color='b')
plt.plot(indices, rmse_esm, marker='o', label='ESM', color='r')
plt.plot(indices, rmse_garch, marker='o', label='GARCH', color='g')
plt.plot(indices, rmse_rf, marker='o', label='RF_Autoreg', color='m')

plt.xlabel('Variable Dependiente')
plt.ylabel('RMSE')
plt.title('Comparación de RMSE entre ARIMA, ESM, GARCH Y RF Autoreg.')
plt.xticks(indices, variables_dependientes)
plt.legend()

plt.grid(True)
plt.tight_layout()
plt.show()
