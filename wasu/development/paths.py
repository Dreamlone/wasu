from pathlib import Path


def get_project_path() -> Path:
    return Path(__file__).parent.parent.parent


def path_to_data_folder():
    return Path(get_project_path(), 'data')


def path_to_examples_folder():
    return Path(get_project_path(), 'examples')


def path_to_plots_folder():
    return Path(get_project_path(), 'examples', 'plots')


def get_models_path() -> Path:
    """ Folder with serialized models """
    return Path(get_project_path(), 'models')
