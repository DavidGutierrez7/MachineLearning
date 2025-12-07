import os
import json
import csv
import pandas as pd
import h2o
import time
from h2o import H2OFrame
from h2o.estimators import H2OStackedEnsembleEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif



h2o.init()

#######################################################################################################################
# Cargamos los datos
training_features_data = pd.read_csv("../datasets_originales/training_set_features.csv", sep=',')
training_set_labels = pd.read_csv("../datasets_originales/training_set_labels.csv", sep=',')
test_features_data = pd.read_csv("../datasets_originales/test_set_features.csv", sep=',')

# Procesamiento de los datos para el TRAIN
# Rellenar las columnas numéricas con la media
numerical_cols = training_features_data.select_dtypes(include=['number']).columns
training_features_data[numerical_cols] = training_features_data[numerical_cols].fillna(value=-1)

# Rellenar las columnas categóricas con un valor por defecto ('out-of-category')
categorical_cols = training_features_data.select_dtypes(include=['object']).columns
training_features_data[categorical_cols] = training_features_data[categorical_cols].fillna('out-of-category')

# Encoding de características categóricas (str --> float)
enc = OrdinalEncoder()
scaler = StandardScaler()
enc.fit(training_features_data)
training_features_data_arr = enc.transform(training_features_data)

col_names_list = training_features_data.columns
encoded_categorical_df = pd.DataFrame(training_features_data_arr, columns=col_names_list)

# Normalización (valores entre 0-1)
scaler.fit(encoded_categorical_df)
normalized_arr = scaler.transform(encoded_categorical_df)
normalized_df = pd.DataFrame(normalized_arr, columns=col_names_list)


# Procesamiento de los datos para el TEST
# Rellenar columnas numéricas
numerical_cols = test_features_data.select_dtypes(include=['number']).columns
test_features_data[numerical_cols] = test_features_data[numerical_cols].fillna(value=-1)

# Rellenar columnas categóricas
categorical_cols = test_features_data.select_dtypes(include=['object']).columns
test_features_data[categorical_cols] = test_features_data[categorical_cols].fillna('out-of-category')

# Encoding de características categóricas (str --> float)
enc.fit(test_features_data)
test_features_data_arr = enc.transform(test_features_data)

col_names_list = test_features_data.columns
test_encoded_categorical_df = pd.DataFrame(test_features_data_arr, columns=col_names_list)

# Normalización (valores entre 0-1)
test_normalized_arr = scaler.transform(test_encoded_categorical_df)
test_normalized_df = pd.DataFrame(test_normalized_arr, columns=col_names_list)

X = normalized_df
y_h1n1 = training_set_labels['h1n1_vaccine']
y_seasonal = training_set_labels['seasonal_vaccine']

# Selección de características
# H1N1
mi_h1n1 = mutual_info_classif(X, y_h1n1, random_state=42)

# Los mejores x features para H1N1
df_h1n1 = pd.DataFrame(mi_h1n1, index=X.columns, columns=['Mutual Information']).sort_values(by='Mutual Information', ascending=False).head(29)
cols_h1n1 = df_h1n1.index

# SEASONAL
mi_seasonal = mutual_info_classif(X, y_seasonal, random_state=42)

# Los mejores features para Seasonal
df_seasonal = pd.DataFrame(mi_seasonal, index=X.columns, columns=['Mutual Information']).sort_values(by='Mutual Information', ascending=False).head(27)
cols_seasonal = df_seasonal.index

X_h1n1_train = normalized_df[cols_h1n1]
X_seasonal_train = normalized_df[cols_seasonal]

# Hacer las predicciones
X_h1n1 = test_normalized_df[cols_h1n1]
X_seasonal = test_normalized_df[cols_seasonal]
#######################################################################################################################



def rebuild_ensemble(model_dir: str) -> H2OStackedEnsembleEstimator:
    """
    Reconstruye un ensemble a partir de los modelos guardados.
    
    Args:
        model_dir (str): Directorio donde están guardados los modelos JSON
    """
    
    # Cargar todos los modelos base
    base_models = []
    metalearner_params = None
    
    # Listar todos los archivos JSON en el directorio
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.json')]
                
    for filename in model_files:
        filepath = os.path.join(model_dir, filename)
        with open(filepath, 'r') as f:
            params = json.load(f)
            
            # Identificar si es un modelo base o el metalearner
            if 'metalearner' in filename:
                metalearner_params = params
            else:
                model_id = filename.replace('.json', '')
                # Eliminamos los parametros nulos para evitar problemas
                params = {k: v for k, v in params.items() if v is not None}
                if 'nfolds' in params:
                    del params['nfolds']

                # Activamos el parámetro 'keep_cross_validation_predictions' si no está presente
                # para poder reconstruir el ensemble
                if 'keep_cross_validation_predictions' in params and params['keep_cross_validation_predictions'] == False:
                    params['keep_cross_validation_predictions'] = True

                if 'balance_classes' in params:
                    del params['balance_classes']

                nfolds = 5
                eval_metric = 'AUC'
                balance_classes = True

                # Reconstruir el modelo base usando sus parámetros
                if filename.startswith('GBM'):
                    model = H2OGradientBoostingEstimator(**params,model_id=model_id,nfolds=nfolds, balance_classes=balance_classes)
                elif filename.startswith('DRF'):
                    model = H2ORandomForestEstimator(**params,model_id=model_id,nfolds=nfolds, balance_classes=balance_classes)
                elif filename.startswith('GLM'):
                    model = H2OGeneralizedLinearEstimator(**params,model_id=model_id,nfolds=nfolds, balance_classes=balance_classes)
                elif filename.startswith('DeepLearning'):
                    model = H2ODeepLearningEstimator(**params,model_id=model_id,nfolds=nfolds, balance_classes=balance_classes)
                elif filename.startswith('XRT'):
                    # Eliminamos el parámetro 'balance_classes' para evitar problemas
                    if 'balance_classes' in params:
                        del params['balance_classes']

                    model = H2ORandomForestEstimator(**params,model_id=model_id,nfolds=nfolds, balance_classes=balance_classes)

                if model_dir.endswith('h1n1_models'):
                    df_h1n1_ = pd.concat([X_h1n1_train, y_h1n1], axis=1)
                    x_column = list(X_h1n1_train.columns)
                    y_column = 'h1n1_vaccine'

                elif model_dir.endswith('seasonal_models'):
                    df_h1n1_ = pd.concat([X_seasonal_train, y_seasonal], axis=1)
                    x_column = list(X_seasonal_train.columns)
                    y_column = 'seasonal_vaccine'

                h2o_df = h2o.H2OFrame(df_h1n1_)
                h2o_df[y_column] = h2o_df[y_column].asfactor()
                # Entrenar el modelo base
                model.train(
                    x=x_column,  
                    y=y_column,   
                    training_frame=h2o_df,
                    max_runtime_secs=100 
                )

                base_models.append(model)
    
    # Reconstruir el metalearner
    if metalearner_params:

        # Borrar el parametro 'lamda'
        if 'lambda' in metalearner_params:
            del metalearner_params['lambda']

        # Borrar los parametros nulos
        metalearner_params = {k: v for k, v in metalearner_params.items() if v is not None}
        
        # Crear el ensemble usando los modelos base y el metalearner
        ensemble = H2OStackedEnsembleEstimator(
            model_id="stacked_ensemble_" + str(int(time.time())),
            base_models=base_models,
            metalearner_algorithm='GLM',
            metalearner_params=metalearner_params
        )

        for model in base_models:
            if model is None:
                print("Modelo base no creado (NONE)")
        
        # Guardar el ensemble en el servidor H2O
        ensemble.train(
            x=x_column,  
            y=y_column,   
            training_frame=h2o_df,
            max_runtime_secs=300
        )

        print("Ensemble reconstruido exitosamente")
        return ensemble
    else:
        print("No se encontró el metalearner")
        return None
            

def predictions(h1n1_ensemble: H2OStackedEnsembleEstimator, seasonal_ensemble: H2OStackedEnsembleEstimator):
    # Parsear los datasets a H2OFrame
    X_h1n1_ = H2OFrame(X_h1n1)
    X_seasonal_ = H2OFrame(X_seasonal)
    predictions_h1n1 = h1n1_ensemble.predict(X_h1n1_)
    predictions_seasonal = seasonal_ensemble.predict(X_seasonal_)

    probabilidad_clase_1 = predictions_h1n1['p1']
    probabilidad_clase_1_lista = probabilidad_clase_1.as_data_frame().values.flatten()

    probabilidad_clase_2 = predictions_seasonal['p1']
    probabilidad_clase_2_lista = probabilidad_clase_2.as_data_frame().values.flatten()

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
    output_path = "predicciones_automl_replica.csv"
    result_df.to_csv(output_path, index=False)


def main():
    h1n1_path = r'models_config/model_params/h1n1_models'
    seasonal_path = r'models_config/model_params/seasonal_models'

    # Creamos el esamble para el modelo Seasonal
    seasonal_ensemble = rebuild_ensemble(seasonal_path)

    # Creamos el esamble para el modelo H1N1
    h1n1_ensemble = rebuild_ensemble(h1n1_path)


    if h1n1_ensemble and seasonal_ensemble:
        print("Ensembles creados exitosamente")

    # Usamos los modelos para hacer predicciones
    predictions(h1n1_ensemble, seasonal_ensemble)

if __name__ == '__main__':
    main()