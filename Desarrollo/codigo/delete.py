import mlflow
import dagshub


dagshub.init(repo_owner='JCOQUE', repo_name='TFG-ingenieria', mlflow=True) 

# Get the ID of the deleted experiment
deleted_experiments = mlflow.tracking.MlflowClient(tracking_uri= 'https://dagshub.com/JCOQUE/TFG-ingenieria.mlflow').search_experiments(view_type=mlflow.entities.ViewType.DELETED_ONLY)
experiment_id = None
for exp in deleted_experiments:
    if exp.name == 'Compras LightGBM':
        experiment_id = exp.experiment_id
        break

if experiment_id:
    print(experiment_id)
    # Permanently delete the experiment
    try:
        mlflow.tracking.MlflowClient(tracking_uri= 'https://dagshub.com/JCOQUE/TFG-ingenieria.mlflow').delete_experiment(experiment_id)
        print(f"Permanently deleted experiment 'Compras LightGBM' with ID {experiment_id}")
    except:
        print(experiment_id)
else:
    print("Experiment 'Compras LightGBM' not found in deleted experiments")