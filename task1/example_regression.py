import mlflow
import numpy as np
from time import sleep
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from ucimlrepo import fetch_ucirepo


# fetch dataset
wine_quality = fetch_ucirepo(id=186)
# data (as pandas dataframes)
X = wine_quality.data.features
y = wine_quality.data.targets

df_wine = X.join(y)
X.drop(['density','free_sulfur_dioxide'], axis = 1, inplace = True)
X = X[['alcohol', 'volatile_acidity', 'total_sulfur_dioxide']]
y['quality'] = y['quality'].astype(float)
# sliptting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # Train different models of regression
    multiple_linear_regression_model()
    decision_tree_model()
    lasso_regression_model()
    ridge_regression_model()

# multiple linear regression model
def multiple_linear_regression_model():
    aux = LinearRegression()
    # Define the model hyperparameters
    param_grid = {'n_jobs': list(range(1, 10)),  # Number of jobs to run in parallel
                  'fit_intercept': [True,False]}  # Testing alpha values from 0.1 to 1.0
    # Set up GridSearchCV for Lineal Regression
    grid_search= GridSearchCV(estimator=aux, param_grid=param_grid, 
                                        scoring='r2', cv=20, n_jobs=-1,verbose=1)

    grid_search.fit(X_train, y_train)

    params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"Best score: {best_score}")
    sleep(5)

    # Predict on the test set
    y_pred = grid_search.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Create a new MLflow Experiment
    mlflow.set_experiment("ML-task1-regression")

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params) # Paso un diccionario de parametros (**params)
        # en caso de que no se haya definido el diccionario de parametros
        # mlflow.log_param("solver", "lbfgs")

        # Log the loss metric - metricas de salida
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Set a tag that we can use to remind ourselves what this run was for
        # añadir tags -> metadatos para identificar el modelo (proyecto, fecha, entidad, empresa,...)
        mlflow.set_tag("Training Info", "Basic Multilineal regression model for wine quality data")

        # Infer the model signature
        # signatura es el esquema que recibe los datos, para este conjuento de entrena y de salida
        # infiereme el modelo de los datos
        signature = infer_signature(X_train, grid_search.predict(X_train))

        # Log the model - imporatnte
        # lo que hace es guardar el modelo
        model_info = mlflow.sklearn.log_model(
            sk_model=grid_search,
            artifact_path="wine_quality_model", 
            signature=signature, # El equema de los datos
            input_example=X_train, # Cuando alguien se baje el ejemplo, le va a dar un ejemplo de como se ve el input
            registered_model_name="multiple_linear_regression", # Nombre del modelo
        )

# Decision Tree model
def decision_tree_model():
    # Define the model hyperparameters with a grid search
    tree_model = DecisionTreeRegressor()

    # Set up the parameter grid
    param_grid = {
        'max_depth': list(range(2, 31)),  # Max depth from 2 to 30
        'min_samples_split': [2, 10, 20, 30, 40],  # Minimum number of samples required to split a node
    }

    # Set up the GridSearchCV with cross-validation and mean squared error as the scoring metric
    grid_search = GridSearchCV(estimator=tree_model,
                            param_grid=param_grid,
                            scoring='r2',  
                            cv=15,  # 15-fold cross-validation
                            n_jobs=-1,  # Use all cores
                            verbose=1)

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Predict on the test set
    y_pred = grid_search.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    params = grid_search.best_params_
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Create a new MLflow Experiment
    mlflow.set_experiment("ML-task1-regression")
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params) # Paso un diccionario de parametros (**params)
        # en caso de que no se haya definido el diccionario de parametros
        # mlflow.log_param("solver", "lbfgs")

        # Log the loss metric - metricas de salida
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Set a tag that we can use to remind ourselves what this run was for
        # añadir tags -> metadatos para identificar el modelo (proyecto, fecha, entidad, empresa,...)
        mlflow.set_tag("Training Info", "Basic Decision Tree model for wine quality data")

        # Infer the model signature
        # signatura es el esquema que recibe los datos, para este conjuento de entrena y de salida
        # infiereme el modelo de los datos
        signature = infer_signature(X_train, grid_search.predict(X_train))

        # Log the model - imporatnte
        # lo que hace es guardar el modelo
        model_info = mlflow.sklearn.log_model(
            sk_model=grid_search,
            artifact_path="wine_quality_model", 
            signature=signature, # El equema de los datos
            input_example=X_train, # Cuando alguien se baje el ejemplo, le va a dar un ejemplo de como se ve el input
            registered_model_name="decision_tree_regression", # Nombre del modelo
        )

# Lasso regression model
def lasso_regression_model():
    aux = LassoCV()
    # Define the model hyperparameters
    alphas = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float64)
    param_grid = {'alphas': [alphas],  # Number of jobs to run in parallel
                  'fit_intercept': [True,False],
                  'max_iter': [1000, 2000,5000, 10000],
                  'tol': [0.0001, 0.001, 0.01, 0.1]}  # Testing alpha values from 0.1 to 1.0
    
    # Set up GridSearchCV for Lineal Regression
    grid_search= GridSearchCV(estimator=aux, param_grid=param_grid, 
                                        scoring='r2', cv=20, n_jobs=-1,verbose=1)

    grid_search.fit(X_train, y_train)

    params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"Best score: {best_score}")
    sleep(5)

    # Predict on the test set
    y_pred = grid_search.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Create a new MLflow Experiment
    mlflow.set_experiment("ML-task1-regression")
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params) # Paso un diccionario de parametros (**params)
        # en caso de que no se haya definido el diccionario de parametros
        # mlflow.log_param("solver", "lbfgs")

        # Log the loss metric - metricas de salida
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Set a tag that we can use to remind ourselves what this run was for
        # añadir tags -> metadatos para identificar el modelo (proyecto, fecha, entidad, empresa,...)
        mlflow.set_tag("Training Info", "Basic Lasso regression model for wine quality data")

        # Infer the model signature
        # signatura es el esquema que recibe los datos, para este conjuento de entrena y de salida
        # infiereme el modelo de los datos
        signature = infer_signature(X_train, grid_search.predict(X_train))

        # Log the model - imporatnte
        # lo que hace es guardar el modelo
        model_info = mlflow.sklearn.log_model(
            sk_model=grid_search,
            artifact_path="wine_quality_model", 
            signature=signature, # El equema de los datos
            input_example=X_train, # Cuando alguien se baje el ejemplo, le va a dar un ejemplo de como se ve el input
            registered_model_name="lasso_regression", # Nombre del modelo
        )

# Ridge regression model
def ridge_regression_model():
    aux = RidgeCV(scoring='r2')
    # Define the model hyperparameters
    param_grid = {'cv': list(range(2, 31)),  
                  'fit_intercept': [True,False],
                  }  # Testing alpha values from 0.1 to 1.0
    
    # Set up GridSearchCV for Lineal Regression
    grid_search= GridSearchCV(estimator=aux, param_grid=param_grid, 
                                        scoring='r2', cv=20, n_jobs=-1,verbose=1)

    grid_search.fit(X_train, y_train)

    params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"Best score: {best_score}")
    sleep(5)

    # Predict on the test set
    y_pred = grid_search.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    # Create a new MLflow Experiment
    mlflow.set_experiment("ML-task1-regression")
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params) # Paso un diccionario de parametros (**params)
        # en caso de que no se haya definido el diccionario de parametros
        # mlflow.log_param("solver", "lbfgs")

        # Log the loss metric - metricas de salida
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Set a tag that we can use to remind ourselves what this run was for
        # añadir tags -> metadatos para identificar el modelo (proyecto, fecha, entidad, empresa,...)
        mlflow.set_tag("Training Info", "Basic Ridge regression model for wine quality data")

        # Infer the model signature
        # signatura es el esquema que recibe los datos, para este conjuento de entrena y de salida
        # infiereme el modelo de los datos
        signature = infer_signature(X_train, grid_search.predict(X_train))

        # Log the model - imporatnte
        # lo que hace es guardar el modelo
        model_info = mlflow.sklearn.log_model(
            sk_model=grid_search,
            artifact_path="wine_quality_model", 
            signature=signature, # El equema de los datos
            input_example=X_train, # Cuando alguien se baje el ejemplo, le va a dar un ejemplo de como se ve el input
            registered_model_name="ridge_regression", # Nombre del modelo
        )

if __name__ == "__main__":
    # Train different models of regression
    main()
