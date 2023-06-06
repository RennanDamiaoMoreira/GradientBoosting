import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

#LEITURA DA BASE

base = pd.read_excel('abalone.xlsx', header=None)

#PRÉ PROCESSAMENTO

# Separar os atributos de entrada (X) e o rótulo alvo (y)
y = base.iloc[:, 0].values
X = base.iloc[:, 1:9].values

LeX = LabelEncoder()
X[:,0] = LeX.fit_transform(X[:,0])

scaler = StandardScaler()
X = scaler.fit_transform(X)

#TREINO 

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Criar o modelo de regressão por Gradient Boosting

#{'squared_error', 'huber', 'quantile', 'absolute_error'}
reg_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42, max_depth= 3)

# Treinar o modelo de regressão
reg_model.fit(X_train, y_train)

# Fazer previsões com o modelo treinado
reg_predictions = reg_model.predict(X_test)

# Avaliar o desempenho do modelo de regressão
mae = mean_absolute_error(y_test, reg_predictions)
print("MAE:", mae)
