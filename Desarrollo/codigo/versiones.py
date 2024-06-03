import subprocess
import pkg_resources

# List of libraries to check
libraries = [
    "keras",
    "tensorflow",
    "xgboost",
    "lightgbm",
    "mlflow",
    "numpy",
    "pandas",
    "matplotlib",
    "torch",
    "prefect",
    "darts",
    "azure-storage-blob",
    "dagshub"
]

def get_library_version(lib_name):
    try:
        version = pkg_resources.get_distribution(lib_name).version
        return version
    except pkg_resources.DistributionNotFound:
        return "Not installed"

def print_library_versions(libraries):
    for lib in libraries:
        version = get_library_version(lib)
        print(f"{lib}: {version}")

print_library_versions(libraries)