import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

#LEITURA DA BASE

base = pd.read_excel('bank.xlsx', header=None)

#PRÉ PROCESSAMENTO

base.drop(base.columns[13], axis=1, inplace=True)

# Separar os atributos de entrada (X) e o rótulo alvo (y)
y = base.iloc[:, 0].values
X = pd.get_dummies(base.iloc[:, 1:20])

# Converter todos os nomes das colunas para o tipo string
X.columns = X.columns.astype(str)

scaler = StandardScaler()
X = scaler.fit_transform(X)

#TREINO 

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Criar o modelo de classificação por Gradient Boosting

#loss{‘log_loss’, ‘deviance’, ‘exponential’}
clf_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, random_state=42, max_depth=3)

# Treinar o modelo de classificação
clf_model.fit(X_train, y_train)

# Fazer previsões com o modelo treinado
clf_predictions = clf_model.predict(X_test)

# Avaliar o desempenho do modelo de classificação
score = accuracy_score(y_test, clf_predictions)
cm = confusion_matrix(y_test,  clf_predictions)
print("Score:", score)
