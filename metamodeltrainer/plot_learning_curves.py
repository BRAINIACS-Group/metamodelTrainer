
#STL imports
from pathlib import Path

#3rd party modules

#local modules
from metamodeltrainer.models import load_model

def get_model_res(model_dir:Path):
    model = load_model(model_dir)
    model.predict()
    

def plot_models(dir_path:Path):
    for model_dir in dir_path.glob('model*'):
        if model_dir.name == "model_final":
            continue
        model_res = get_model_res(model_dir)
        model_list.append()
    pass

if __name__ == "__main__":
    cur_dir = Path(__file__).reolve().parents[1]
    plot_models(cur_dir / "models_20231116")