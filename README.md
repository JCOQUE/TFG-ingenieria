# Intro

In this README file, important information about the project will be described such as: links where the core information has been found, important aspect to keep in mind when modifying the code and other things. All this will be organized by tool used in the project such as PowerBI, Azure, etc.

## Tools
In this sections, the software tools, along with some information source will be discussed.
### Mlflow
Mlflow is an MLOPS tool that allows you to save a model, its parameters, metrics and anything else related to the model (.csv, .png, etc.) when training. This allows you to track every model that has been trained, and keep their information. This allows you to reuse this model whenever you want. It also allows you to compare different models based on the metrics saved. Mlflow has 'experiments'. Each experiment should save different trained models of an algorithm. In this project, since 4 algorithms were applied and for two targets (Compras and Ventas), I have a total of 8 experiments. 
- No specific source info was searched for this tool. Anything on internes is helpful.
### Dagshub
In a nutshell, Dagshub is a github for ML and data engineers. The main reason to use Dagshub in this project, is to be able to deploy Mlflow in a non-local environment so that it simulates in a better way a real case scenario. 
###### Related info sources for this project:
- https://www.youtube.com/watch?v=K9se7KQON5k&t=695s&pp=ygUTbWxmbG93IHdpdGggZGFnc2h1Yg%3D%3D

## Prefect
For training automation, Prefect was used. Even though python files can be executed automatically with cron jobs that every OS has, there is now way to track the execution. Prefect allows you to automate file execution, setting input parameters to this file, set some execution policies such as: retries, maximum amount of time to execute a file to save computer resources and much more. It also provides a web where you can track the executions of you files, if it failed, why it failed. Additionally it integrates with other software tools in case you are using them in you code. This way, you can save in prefect api keys, access tokens, and other security stuff that you use in you code.
###### Related info sources for this project:
- Good Prefect intro:  https://www.youtube.com/watch?v=D5DhwVNHWeU&t=1126s&pp=ygUHcHJlZmVjdA%3D%3D
> Note: Airflow was also considered. You can use that software tool instead if prefered.

### Azure
Azure is a cloud computing platform and service created by Microsoft. 
#### Azure Blob Storage
In this project, Azure Blob Storage is a datalake used for two things: raw .csv used in this project is saved here. And the transformed (added columns, predictions and more) .csv is also saved here as a .parquet. This .parquet is the one used in PowerBI for data visualization.
#### Data Factory
Data Factory (a.k.a ADF (Azure Data Factory) is another resource provided by Azure that allows you to create pipelines to move, transform and process data between different sources. For this project, it was used to copy data from the .parquet to an SQL Database.
#### Azure SQL Database
The Azure SQL Database is the source from which PowerBI is going to retrieve data from. The main reason to use SQL Database as the PowerBI source is because it is a source that allows query folding for incremental refreshing in PowerBI (see PowerBI section).
> Note: The reason to have used Azure Blob Storage and Azure SQL Database is because it simulates a real case scenario much better where the .csv and the SQL Database are decoupled.
###### Related info sources for this project:
- Video 1: https://www.youtube.com/watch?v=xBJbvTAi5lY

- Video 2: https://www.youtube.com/watch?v=gc5mWkRPfWM

- Video3: https://www.youtube.com/watch?v=07A3LPfiu18

**Additionally to that**:
- How to perform UPSERT with Copy Data activity: https://www.youtube.com/watch?v=fegEN1Z1viM
- Datatype conversion: https://www.youtube.com/watch?v=vB446EB_-aU


### PowerBI
PowerBI is a data visualization tool. There is no specific reason to have chosen PowerBI over Tableau. [Zebra BI](https://zebrabi.com/pbi-pro-trial/?utm_term=zebra%20power%20bi&utm_campaign=Brand+campaign&utm_source=google&utm_medium=cpc&hsa_acc=5445927020&hsa_cam=10896260820&hsa_grp=108906923204&hsa_ad=634945809310&hsa_src=g&hsa_tgt=kwd-930008850410&hsa_kw=zebra%20power%20bi&hsa_mt=b&hsa_net=adwords&hsa_ver=3&gad_source=1&gclid=Cj0KCQjwpNuyBhCuARIsANJqL9ObqxkQ7e0maiaNObgA4knPiEXp2ruSty7l4JRZHvW-8eI3nH1wF6oaAignEALw_wcB), and [PureViz](https://pureviz.net/infographic) graphics were used.
###### Related info sources for this project:
- Incremental Refreshing: https://www.youtube.com/watch?v=Kui_1G6kQIQ
- PowerBI AI features: https://www.youtube.com/watch?v=5URcVnV29Is&list=PLa370_-deBCQOtfe2OpxuMJZ9C1RC0YJf







