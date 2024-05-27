from prefect import flow
from prefect.runner.storage import GitRepository
from prefect.blocks.system import Secret


# DON'T WORRY ABOUT THE OUTPUT. THE DEPLOYMENT GETS CREATED.
# THERE IS AN ISSUE WITH DEPENDENCIES BETWEEN DUGSHUB AND PREFECT.
# CONCRETLY WITH THE LIBRARY RICH
if __name__ == "__main__":
    flow.from_source(
    source=  "https://github.com/JCOQUE/TFG-ingenieria.git",
    entrypoint="Desarrollo/codigo/LightGBM.py:hello_world",
    ).deploy(
    name="LightGBM Compras",
    work_pool_name="TFGinso-work-pool"#,
    #cron="*/3 * * * *",
    )

