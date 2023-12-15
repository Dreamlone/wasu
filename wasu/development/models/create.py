from pathlib import Path

from wasu.development.paths import get_models_path


class ModelsCreator:
    """ Class allow to create paths for model """

    def __init__(self, name: str):
        self.name = name

    def model_folder(self):
        path = Path(get_models_path(), self.name)
        path = path.resolve()

        path.mkdir(exist_ok=True, parents=True)

        return path
