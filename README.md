# ARI5121 (Applied NLP) Speech Assignment

The purpose of this GitHub repository is to host any code required for the ARI5121 (Speech) Applied Natural Language Processing study-unit assignment.

## Project notes:

- **Python version used**: 3.12.3.
- **Packages used**: Refer to packages inside `requirements.txt`.

## Replication of virtual environment (.venv)

Assuming that Python is already pre-installed on the host machine.

1. Run `python -m venv .venv` to create the Python virtual environment. `python` here refers to the alias of the Python executable path and depends on the alias used on the host machine (full Python path can also be used). Running this command will create a Python virtual environment depending on the base Python version being used to create the environment.
2. Activate virtual environment by running `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux).
3. Upgrade `pip` (Python's package manager) by running `pip install --upgrade pip`.
4. Run `pip install -r requirements.txt` to download any packages required for this project.

## Project Directory

- `datasets/`: Contains all .wav files used for this assignment. Dataset is from the Accents of the British Isles (ABI-1) Corpus.
- `images/`: Contains images of charts/assets created throughout the assignment.
- `papers/`: Contains PDFs of papers referenced during the assignment.
- `.gitignore`: Git file to ignore certain files from being source-controlled.
- `cleanup_dataset.py`: Python script to cleanup the ABI-1 Corpus to only keep .wav files required for the assignment.
- `comparisons.pt`: PyTorch tensors for storing all comparisons between embeddings. Comparisons refer to cosine similarity between one embedding and another.
- `core.ipynb`: Jupyter notebook that contains all the code & analysis done for this assignement (except for the cleaning of the dataset).
- `embeddings.pt`: PyTorch tensors for storing all embeddings.
- `README.md`: Project guide.
- `requirements.txt`: Text file denoting list of packages used for this assignment.