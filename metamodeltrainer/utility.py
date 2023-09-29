
#STL imports

#3rd party imports
import pandas as pd

#pyVlab imports
from pyVlab import Geometry

def convert_to_stress(dataset:pd.DataFrame,geom:Geometry)->pd.DataFrame:
    '''convert to stress and strain values and return converted dataframe
    Args:
        dataset: pandas Dataframe with force, displacement, angle and torque
        geom: Geometry object used for the conversion
    Returns:
        converted Dataframe
    Raises:
    '''
    dataset_stress = pd.DataFrame({
        'time':dataset.time,
        'stretch' : geom.get_axialstretch(dataset.displacement),
        'normal_stress' : geom.get_axialstress(dataset.force),
        'shear': geom.get_torsionstrain(dataset.angle),
        'shear_stress':geom.get_torsionstress(dataset.torque),
    })
    return dataset_stress

def convert_to_force_disp(dataset:pd.DataFrame,geom:Geometry)->pd.DataFrame:
    '''convert to force, displacement, angle and torque values
    Args:
        dataset: pandas Dataframe with stress and strain values
        geom: geometry object used for the conversion
    Returns:
        converted Dataframe
    Raises:'''
    dataset_force_disp = pd.DataFrame({
        'time': dataset.time,
        'displacement':geom.get_axialdisplacement_from_stretch(dataset.stretch),
        'force': geom.get_axialforce_from_stress(dataset.normal_stress),
        'angle': geom.get_torsion_ang_displacement_from_strain(dataset.shear),
        'torque': geom.get_torque_from_stress(dataset.shear_stress),
    })
    return dataset_force_disp