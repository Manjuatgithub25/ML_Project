import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s:] %(message)s:')

project_version = '0.0.1'


list_of_files = [
    # ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/logging/__init__.py",
    "src/config/__init__.py",
    "src/config/configuration.py",
    "src/entity/__init__.py",
    "src/constants/__init__.py",
    "src/pipeline/__init__.py",
    "src/utils/__init__.py",
    "src/utils/common.py",
    "confiig/config.yaml",
    "reasearch/trials.ipynb",
    "params.yaml",
    "main.py",
    "setup.py",
    "app.py"
]


for file in list_of_files:
    file_path  = Path(file)
    file_dir, file_name = os.path.split(file)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"{file_dir} has been created for the file {file_name}")

    if os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass
            logging.info(f"creating empty file :{file_path}")
    else:
        logging.info(f"File {file_name} already exists")