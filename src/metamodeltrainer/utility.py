
#STL imports

#3rd party imports
import pandas as pd

#pyVlab imports
from vlab_utilities import Geometry

def convert_to_stress(dataset:pd.DataFrame,geom:Geometry)->pd.DataFrame:
    '''convert to stress and strain values and return converted dataframe
    Args:
        dataset: pandas Dataframe with force, displacement, angle and torque
        geom: Geometry object used for the conversion
    Returns:
        converted Dataframe
    Raises:
    '''
    col_func_list = {
            ('time','time',lambda s: s),
            ('displacement','stretch',geom.get_axialstretch),
            ('force','normal_stress',geom.get_axialstress),
            ('angle','shear', geom.get_torsionstrain),
            ('torque','shear_stress',geom.get_torsionstress),
        }
    stress_col_dict = {} 
    for colname_old,colname_new,col_fn in col_func_list:
        if colname_old in dataset:
            stress_col_dict[colname_new] = col_fn(dataset[colname_old])
            
    dataset_stress = pd.DataFrame(stress_col_dict)
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