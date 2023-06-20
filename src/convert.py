from models import *
from pathlib import Path
import os

def convert(name):
    model = load_model(name)
    model.newsave(Path(output_dir,name[:-4]))

input_dir = r"/data/iwtm662/efiNeuralNetwork/models"
output_dir = r"/data/iwtm662/efiNeuralNetwork/new_models"

for f in os.listdir(input_dir):
    if f.endswith('.pkl'):
        convert(f)
