import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# Leer datos
training_features_data = pd.read_csv("Machine_Learning_Lab/task2/datasets_originales/training_set_features.csv", sep=',')
training_set_labels = pd.read_csv("Machine_Learning_Lab/task2/datasets_originales/training_set_labels.csv", sep=',')
test_features_data = pd.read_csv("Machine_Learning_Lab/task2/datasets_originales/test_set_features.csv", sep=',')

print(test_features_data.shape)  
print(training_set_labels.shape) 


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Preprocesamiento para training_features_data------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

# Eliminar valores nulos en training_features_data

# Rellenar las columnas numéricas con la media
numerical_cols = training_features_data.select_dtypes(include=['number']).columns
training_features_data[numerical_cols] = training_features_data[numerical_cols].fillna(training_features_data[numerical_cols].mean())

# Rellenar las columnas categóricas con un valor por defecto ('out-of-category')
categorical_cols = training_features_data.select_dtypes(include=['object']).columns
training_features_data[categorical_cols] = training_features_data[categorical_cols].fillna('out-of-category')

# Verificar que no queden valores nulos
print(training_features_data.isna().sum())

# Encoding de características categóricas (str --> float)
enc = OrdinalEncoder()
enc.fit(training_features_data)
training_features_data_arr = enc.transform(training_features_data)

col_names_list = training_features_data.columns
encoded_categorical_df = pd.DataFrame(training_features_data_arr, columns=col_names_list)

# Normalización (valores entre 0-1)
scaler = StandardScaler()
scaler.fit(encoded_categorical_df)
normalized_arr = scaler.transform(encoded_categorical_df)
normalized_df = pd.DataFrame(normalized_arr, columns=col_names_list)

print(normalized_df.info())

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Preprocesamiento para test_features_data ------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

# Eliminar valores nulos en test_features_data

# Rellenar columnas numéricas
numerical_cols = test_features_data.select_dtypes(include=['number']).columns
test_features_data[numerical_cols] = test_features_data[numerical_cols].fillna(test_features_data[numerical_cols].mean())

# Rellenar columnas categóricas
categorical_cols = test_features_data.select_dtypes(include=['object']).columns
test_features_data[categorical_cols] = test_features_data[categorical_cols].fillna('out-of-category')

# Verificar que no queden valores nulos
print(test_features_data.isna().sum())

# Encoding de características categóricas (str --> float)
enc = OrdinalEncoder()
enc.fit(test_features_data)
test_features_data_arr = enc.transform(test_features_data)

col_names_list = test_features_data.columns
test_encoded_categorical_df = pd.DataFrame(test_features_data_arr, columns=col_names_list)

# Normalización (valores entre 0-1)
test_normalized_arr = scaler.transform(test_encoded_categorical_df)
test_normalized_df = pd.DataFrame(test_normalized_arr, columns=col_names_list)


#-----------------------------------------------------------------------------------
# Datos -----------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

X = normalized_df
y_h1n1 = training_set_labels['h1n1_vaccine']
y_seasonal = training_set_labels['seasonal_vaccine']

CV_h1_n1 = StratifiedShuffleSplit(n_splits=5, random_state = 42)
CV_seasonal = StratifiedKFold(n_splits=15, random_state=42, shuffle=True)

#-----------------------------------------------------------------------------------
# Mutual Information para H1N1
#-----------------------------------------------------------------------------------
mi_h1n1 = mutual_info_classif(X, y_h1n1, random_state=42)

# Los mejores 8 features para H1N1
df_h1n1 = pd.DataFrame(mi_h1n1, index=X.columns, columns=['Mutual Information']).sort_values(by='Mutual Information', ascending=False).head(13)
print(df_h1n1)

# Selección de características para H1N1
X_h1n1 = X[df_h1n1.index]

#------------------------------------------------------------------------------------------------------------------
# Random Forest Classifier for H1N1  ------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

cat_model = CatBoostClassifier(eval_metric='AUC', random_seed=42, verbose=0)
rf_model = RandomForestClassifier(random_state=42, class_weight={0: 0.55, 1: 0.45})
gb_model = GradientBoostingClassifier(random_state=42)
xgb_model = XGBClassifier(random_state=42)
voting_clf = VotingClassifier(estimators=[('rf', rf_model), ('gb', gb_model), ('xgb', xgb_model)], voting='soft')

param_grid = {

    'rf__n_estimators': [300],
    'rf__max_depth': [10],
    'rf__min_samples_split': [55],
    'rf__min_samples_leaf': [4],
    'rf__bootstrap': [False],


    'gb__n_estimators': [350],
    'gb__learning_rate': [0.1],
    'gb__min_samples_split': [10],
    'gb__subsample': [0.8],
    'gb__loss': ['log_loss'],


    'xgb__n_estimators': [100],
    'xgb__learning_rate': [0.1], # Tasa de aprendizaje
    'xgb__max_depth': [5],  # Profundidad máxima de los árboles
    'xgb__subsample': [0.6], # Proporción de datos utilizada en cada árbol
    'xgb__colsample_bytree': [0.6], # Proporción de características por árbol
    'xgb__gamma': [0],  # Reducción mínima en pérdida requerida para dividir
    'xgb__min_child_weight': [1], # Peso mínimo de las hojas para evitar divisiones muy pequeñas
    'xgb__scale_pos_weight': [1] # Compensación de clases desbalanceadas

}


grid = GridSearchCV(estimator=voting_clf, param_grid=param_grid, cv=CV_h1_n1, n_jobs=-1, scoring='roc_auc')

# Entrenar el modelo para H1N1
grid.fit(X_h1n1, y_h1n1)

print("The best parameters are %s with a score of %0.4f\n"
      % (grid.best_params_, grid.best_score_))

# Obtener las predicciones para el dataset de test
predicciones = grid.predict_proba(test_normalized_df[df_h1n1.index])

#-----------------------------------------------------------------------------------
# Mutual Information para Seasonal
#-----------------------------------------------------------------------------------

mi_seasonal = mutual_info_classif(X, y_seasonal, random_state=42)

# Los mejores 8 features para Seasonal
df_seasonal = pd.DataFrame(mi_seasonal, index=X.columns, columns=['Mutual Information']).sort_values(by='Mutual Information', ascending=False).head(17)
print(df_seasonal)

# Selección de características para Seasonal
X_seasonal = X[df_seasonal.index]

#------------------------------------------------------------------------------------------------------------------
# Random Forest Classifier for Seasonal  ---------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
cat_model = CatBoostClassifier(eval_metric='AUC', random_seed=42, verbose=0)
xgb_model = XGBClassifier(random_state=42)
voting_clf = VotingClassifier(estimators=[('ct', cat_model), ('gb', gb_model), ('xgb', xgb_model)], voting='soft')

param_grid = {

    'ct__iterations': [1500],
    'ct__learning_rate': [0.01],
    'ct__depth': [6],
    'ct__l2_leaf_reg': [5],
    'ct__border_count': [32],
    'ct__bagging_temperature': [7],

    'gb__n_estimators': [350],
    'gb__learning_rate': [0.1],
    'gb__min_samples_split': [10],
    'gb__subsample': [0.8],
    'gb__loss': ['log_loss'],

    'xgb__n_estimators': [300], 
    'xgb__learning_rate': [0.1],
    'xgb__max_depth': [3], 
    'xgb__subsample': [0.6],
    'xgb__colsample_bytree': [0.6],
    'xgb__gamma': [0],  
    'xgb__min_child_weight': [1], 
    'xgb__scale_pos_weight': [1] 


}

grid = GridSearchCV(estimator=voting_clf, param_grid=param_grid, cv=CV_seasonal, n_jobs=-1, scoring='roc_auc')

# Entrenar el modelo para Seasonal
grid.fit(X_seasonal, y_seasonal)

print("The best parameters are %s with a score of %0.4f\n"
        % (grid.best_params_, grid.best_score_))

# Obtener las predicciones para el dataset de test
predicciones2 = grid.predict_proba(test_normalized_df[df_seasonal.index])


#'''
#------------------------------------------------------------------------------------------------------------
# Extraer las probabilidades de la clase 1
h1n1_class_1 = predicciones[:, 1]  # Clase 1 para h1n1_vaccine
seasonal_class_1 = predicciones2[:, 1]  # Clase 1 para seasonal_vaccine

# Generar los IDs empezando en 26707
respondent_ids = range(26707, 26707 + len(h1n1_class_1))  # Generar una secuencia de IDs

# Crear el DataFrame con el formato solicitado
result_df = pd.DataFrame({
    'respondent_id': respondent_ids,  # IDs generados secuencialmente
    'h1n1_vaccine': h1n1_class_1,
    'seasonal_vaccine': seasonal_class_1
})

# Guardar el DataFrame como un archivo CSV con los valores en formato de un solo decimal
output_path = "predicciones_final2.csv"
result_df.to_csv(output_path, index=False)  # Sin especificar float_format, se guardan todos los decimales
#'''