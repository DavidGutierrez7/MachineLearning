import csv
import os
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from sklearn.preprocessing import OrdinalEncoder as oe
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import VotingClassifier
from sklearn.utils import class_weight
from xgboost import XGBClassifier
import h2o
from h2o.automl import H2OAutoML
from h2o.frame import H2OFrame
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs



class InfoExtractor:

    def __init__(self):
        pass

   
    def extract_model_info(self,json_data: Dict) -> List[Dict]:
        """
        Extracts model information from the JSON configuration file.
        
        Args:
            json_data (Dict): The loaded JSON configuration data
            
        Returns:
            List[Dict]: List of dictionaries containing model information
        """
        models = []
        
        # Extract base models from the configuration
        base_models = json_data.get('base_models', {}).get('actual', [])
        
        # Get global parameters that apply to all models
        global_params = {
            'metalearner_algorithm': json_data.get('metalearner_algorithm', {}).get('actual'),
            'metalearner_transform': json_data.get('metalearner_transform', {}).get('actual'),
            'metalearner_nfolds': json_data.get('metalearner_nfolds', {}).get('actual'),
            'max_runtime_secs': json_data.get('max_runtime_secs', {}).get('actual'),
            'seed': json_data.get('seed', {}).get('actual'),
            'response_column': json_data.get('response_column', {}).get('actual', {}).get('column_name')
        }
        
        # Process each base model
        for model in base_models:
            model_info = {
                'model_name': model.get('name'),
                'model_type': model.get('type'),
                'global_parameters': global_params
            }
            models.append(model_info)
            
        
        return models

    
    def save_model_params(self,models: List[Dict], output_dir: str, aml_leader) -> None:
        """
        Saves individual model parameters to separate JSON files.
        
        Args:
            models (List[Dict]): List of model information dictionaries
            output_dir (str): Directory where JSON files will be saved
        """

        # Check if the directory exists
        if os.path.exists(output_dir):
            # If it exists, remove all files inside it
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')


        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        metalearner = aml_leader.metalearner()
        
        # Obtener la tabla de importancia de variables
        varimp = metalearner.varimp()
        
        # Filtrar solo los modelos con importancia > 0
        real_models_used = [model[0] for model in varimp if model[1] > 0]
        
        # Save each model's parameters to a separate file
        for model in real_models_used:
            filename = f"{model}.json"
            filepath = os.path.join(output_dir, filename)
            # Extract parameters from de model
            model_h2o = h2o.get_model(model)
            with open(filepath, 'w') as f:
                json.dump(model_h2o.get_params(), f, indent=4)

        # Guardar parámetros del metalearner
        try:
            metalearner_path = os.path.join(output_dir, "metalearner.json")
            with open(metalearner_path, 'w') as f:
                json.dump(metalearner.get_params(), f, indent=4)
            print("Guardados parámetros del metalearner") 
        except Exception as e:
            print(f"Error: {e}")

    
    def process_vaccine_models(self,json_path: str, output_dir: str, aml_leader) -> None:
        """
        Process both H1N1 and seasonal vaccine model configurations.
        
        Args:
            json_path (str): Path to model JSON file
            output_dir (str): Directory where model parameters will be saved
        """
        # Process models
        with open(json_path, 'r') as f:
            data = json.load(f)
            models = self.extract_model_info(data)

        if json_path == 'models_config/model_params_h1n1.json':
            self.save_model_params(models, os.path.join(output_dir, 'h1n1_models'), aml_leader)
        if json_path == 'models_config/model_params_seasonal.json':
            self.save_model_params(models, os.path.join(output_dir, 'seasonal_models'), aml_leader)
    

ejecutar_h2o = True
guardar_modelos = True

# Inicializar h2o
if ejecutar_h2o:
   h2o.init()
   InfoExtractor = InfoExtractor()
   


# Leer datos
training_features_data = pd.read_csv("../datasets_originales/training_set_features.csv", sep=',')
training_set_labels = pd.read_csv("../datasets_originales/training_set_labels.csv", sep=',')
test_features_data = pd.read_csv("../datasets_originales/test_set_features.csv", sep=',')

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Preprocesamiento para training_features_data------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

# Eliminar valores nulos en training_features_data

# Rellenar las columnas numéricas con la media
numerical_cols = training_features_data.select_dtypes(include=['number']).columns
training_features_data[numerical_cols] = training_features_data[numerical_cols].fillna(value=-1)

# Rellenar las columnas categóricas con un valor por defecto ('out-of-category')
categorical_cols = training_features_data.select_dtypes(include=['object']).columns
training_features_data[categorical_cols] = training_features_data[categorical_cols].fillna('out-of-category')

# Encoding de características categóricas (str --> float)
enc = oe()
enc.fit(training_features_data)
training_features_data_arr = enc.transform(training_features_data)

col_names_list = training_features_data.columns
encoded_categorical_df = pd.DataFrame(training_features_data_arr, columns=col_names_list)

# Normalización (valores entre 0-1)
scaler = StandardScaler()
scaler.fit(encoded_categorical_df)
normalized_arr = scaler.transform(encoded_categorical_df)
normalized_df = pd.DataFrame(normalized_arr, columns=col_names_list)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# Preprocesamiento para test_features_data ------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

# Eliminar valores nulos en test_features_data

# Rellenar columnas numéricas
numerical_cols = test_features_data.select_dtypes(include=['number']).columns
test_features_data[numerical_cols] = test_features_data[numerical_cols].fillna(value=-1)

# Rellenar columnas categóricas
categorical_cols = test_features_data.select_dtypes(include=['object']).columns
test_features_data[categorical_cols] = test_features_data[categorical_cols].fillna('out-of-category')


# Encoding de características categóricas (str --> float)
enc = oe()
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

#-----------------------------------------------------------------------------------
# Feature Selection para H1N1
#-----------------------------------------------------------------------------------
mi_h1n1 = mutual_info_classif(X, y_h1n1, random_state=42)

# Los mejores x features para H1N1
df_h1n1 = pd.DataFrame(mi_h1n1, index=X.columns, columns=['Mutual Information']).sort_values(by='Mutual Information', ascending=False).head(29)

cols_h1n1 = df_h1n1.index
# Selección de características para H1N1
X_h1n1 = X[cols_h1n1]

if ejecutar_h2o:
    df_h1n1_ = pd.concat([X_h1n1, y_h1n1], axis=1)

    h2o_df = h2o.H2OFrame(df_h1n1_)
    # Convertir la columna objetivo a categórica
    h2o_df["h1n1_vaccine"] = h2o_df["h1n1_vaccine"].asfactor()

    aml = H2OAutoML(max_runtime_secs=600, 
                    seed=42, 
                    sort_metric="AUC", 
                    balance_classes=True, 
                    nfolds=10,
                    keep_cross_validation_predictions=True)
    
    aml.train(x=list(X_h1n1.columns), y='h1n1_vaccine', training_frame=h2o_df)
    lb = aml.leaderboard
    print(lb)  # Muestra solo el ID del modelo y el AUC

    # Obtener las predicciones para el dataset de test
    test_h2o_df = h2o.H2OFrame(test_normalized_df[cols_h1n1])

    best_model = aml.leader
    # Metricas de desempeño
    perf = best_model.model_performance(test_data=h2o_df)
    # Matriz de confusión
    print(perf.confusion_matrix())
    # Curva ROC
    perf.plot()
    # F1-Score
    print(perf.metric('f1'))
    

    predicciones = aml.leader.predict(test_h2o_df)
    probabilidad_clase_1 = predicciones['p1']
    probabilidad_clase_1_lista = probabilidad_clase_1.as_data_frame().values.flatten()
    if guardar_modelos:
        print('[H1N1] recuperamos los parametros del modelo')
        with open ('models_config/model_params_h1n1.json', 'w') as f:
            json.dump(aml.leader.params, f, indent=4)
        with open ('models_config/configuracion_modelo_h1n1.txt', 'w') as f:
            f.write(str(aml.leader.summary()))

        aml_leader = aml.leader
        # Guardamos los modelos que se han creado en la sesion
        InfoExtractor.process_vaccine_models('models_config/model_params_h1n1.json', 'models_config/model_params', aml_leader) 


#-----------------------------------------------------------------------------------
# Feature Selection para Seasonal
#-----------------------------------------------------------------------------------

mi_seasonal = mutual_info_classif(X, y_seasonal, random_state=42)

# Los mejores 8 features para Seasonal
df_seasonal = pd.DataFrame(mi_seasonal, index=X.columns, columns=['Mutual Information']).sort_values(by='Mutual Information', ascending=False).head(29)
print(df_seasonal)

cols_seasonal = df_seasonal.index
# Selección de características para Seasonal
X_seasonal = X[cols_seasonal]


if ejecutar_h2o:
    df_seasonal_ = pd.concat([X_seasonal, y_seasonal], axis=1)
    h2o_df = h2o.H2OFrame(df_seasonal_)
    # Convertir la columna objetivo a categórica
    h2o_df["seasonal_vaccine"] = h2o_df["seasonal_vaccine"].asfactor()

    aml = H2OAutoML(max_runtime_secs=600, 
                    seed=42, 
                    sort_metric="AUC", 
                    nfolds=5,
                    keep_cross_validation_predictions=True)
    
    aml.train(x=list(X_seasonal.columns), y='seasonal_vaccine', training_frame=h2o_df)
    lb = aml.leaderboard
    print(lb)  # Muestra solo el ID del modelo y el AUC

    best_model = aml.leader
    # Metricas de desempeño
    perf = best_model.model_performance(test_data=h2o_df)
    # Matriz de confusión
    print(perf.confusion_matrix())
    # Curva ROC
    perf.plot()
    # F1-Score
    print(perf.metric('f1'))

    # Obtener las predicciones para el dataset de test
    test_h2o_df = h2o.H2OFrame(test_normalized_df[cols_seasonal])
    predicciones = aml.leader.predict(test_h2o_df)
    probabilidad_clase_2 = predicciones['p1']
    probabilidad_clase_2_lista = probabilidad_clase_2.as_data_frame().values.flatten()
    if guardar_modelos:
        print('[SEASONAL] recuperamos los parametros del modelo')
        with open ('models_config/model_params_seasonal.json', 'w') as f:
            json.dump(aml.leader.params, f, indent=4)
        with open ('models_config/configuracion_modelo_seasonal.txt', 'w') as f:
            f.write(str(aml.leader.summary()))

        # Guardamos los modelos que se han creado en la sesion
        aml_leader = aml.leader
        InfoExtractor.process_vaccine_models('models_config/model_params_seasonal.json', 'models_config/model_params', aml_leader) 

    print('Finalizando la sesion de h2o')
    h2o.shutdown(prompt=False)
    


#'''
#------------------------------------------------------------------------------------------------------------
# Extraer las probabilidades de la clase 1
h1n1_class_1 = probabilidad_clase_1_lista # Clase 1 para h1n1_vaccine
seasonal_class_1 = probabilidad_clase_2_lista # Clase 1 para seasonal_vaccine

# Generar los IDs empezando en 26707
respondent_ids = range(26707, 26707 + len(h1n1_class_1))  # Generar una secuencia de IDs

# Crear el DataFrame con el formato solicitado
result_df = pd.DataFrame({
    'respondent_id': respondent_ids,  # IDs generados secuencialmente
    'h1n1_vaccine': h1n1_class_1,
    'seasonal_vaccine': seasonal_class_1
})

# Guardar el DataFrame como un archivo CSV con los valores en formato de un solo decimal
output_path = "predicciones_automl.csv"
result_df.to_csv(output_path, index=False)
