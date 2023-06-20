from models import *
from pathlib import Path
import os

def convert(name):
    model = load_model(Path(input_dir,name))
    model.new_save(Path(output_dir,name[:-4]))

input_dir = r"C:\Users\ferla\OneDrive\Documents\GitHub\efiNeuralNetwork\models"
output_dir = r"C:\Users\ferla\OneDrive\Documents\GitHub\efiNeuralNetwork\new_models"

failures = []

for f in os.listdir(input_dir):
    if f.endswith('.pkl'):
        print(f)
        convert(f)
        #except: failures.append(f)

print(failures)