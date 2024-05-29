## Intro

In this README file, important information about the project will be described such as: links where the core information has been found, important aspect to keep in mind when modifying the code and other things. All this will be organized by tool used in the project such as PowerBI, Azure, etc.

## Mlflow
Mlflow is an MLOPS tool that allows you to save a model, its parameters, metrics and anything else related to the model (.csv, .png, etc.) when training. This allows you to track every model that has been trained, and keep their information. This allows you to reuse this model whenever you want. It also allows you to compare different models based on the metrics saved. Mlflow has 'experiments'. Each experiment should save different trained models of an algorithm. In this project, since 4 algorithms were applied and for two targets (Compras and Ventas), I have a total of 8 experiments. 
- No specific source info was searched for this tool. Anything on internes is helpful.
## Dagshub
In a nutshell, Dagshub is a github for ML and data engineers. The main reason to use Dagshub in this project, is to be able to deploy Mlflow in a non-local environment so that it simulates in a better way a real case scenario. 
- Related info source for this project: https://www.youtube.com/watch?v=K9se7KQON5k&t=695s&pp=ygUTbWxmbG93IHdpdGggZGFnc2h1Yg%3D%3D

## Prefect
For training automation, Prefect was used. Even though python files can be executed automatically with cron jobs that every OS has, there is now way to track the execution. Prefect allows you to automate file execution, setting input parameters to this file, set some execution policies such as: retries, maximum amount of time to execute a file to save computer resources and much more. It also provides a web where you can track the executions of you files, if it failed, why it failed. Additionally it integrates with other software tools in case you are using them in you code. This way, you can save in prefect api keys, access tokens, and other security stuff that you use in you code.
- Good Prefect intro:  https://www.youtube.com/watch?v=D5DhwVNHWeU&t=1126s&pp=ygUHcHJlZmVjdA%3D%3D

## Azure
Azure is a cloud computing platform and service created by Microsoft. 
#### Azure Blob Storge
In this project, Azure Blob Storage is used for two things: raw .csv used in this project is saved here. And the transformed (added columns, predictions and more) .csv is also saved here as a .parquet. 
#### Data Factory

