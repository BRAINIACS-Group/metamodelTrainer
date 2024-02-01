
#STL imports
from __future__ import annotations
from pathlib import Path
from typing import List
import logging
import tempfile

#3rd party imports
import numpy as np

#vlab imports
from vlab_utilities import ParameterHandler, OptVars, Simulation

#local imports
from .models import load_model
from .explore_param_space import Sample

class MetaModel(Simulation):

    def __init__(self,model_path,prm:ParameterHandler,out_dir:Path = None) -> None:
        '''create object of type simulation and init properties
        that are shared between all inheriting classes'''
        
        self.model = load_model(model_path)
        self.__prm = prm
        super()._init_from_prm(prm)
        self.out_dir = out_dir
        self.input_data      = self.get_input_disp_data()
        self._input_dir = Path(prm.get_entry('simulation/global',
            'input directory'))
        

    def predict(self,opt_vars:OptVars,outdir:Path,get_jac:bool=False)->None:
        '''Use metamodel to predict output
        
        Args:
            opt_vars: OptVars object representing the variable values
            outdir: Path object representing the output folder
            
        Returns:
        
        Raises:
        '''
        model_parameters = self.model.X_T.columns[:self.model.X_T.p]
        opt_vars_input = [opt_vars[p].value for p in model_parameters]
        sample = Sample([opt_vars_input],model_parameters)
        jac_data = self.model.run(sample,input_dir=self._input_dir,output_dir=outdir,
            parameter_file=self.__prm.filepath,return_jac=get_jac)
        
        if get_jac:
            print('jac shape:',jac_data.jac.shape)
            assert jac_data.jac.shape[0] == 1
            ov_idx_list = [jac_data.param_cols.index(ovn) for ovn in model_parameters]
            print('ov idx list:',ov_idx_list)
            print('jac_data.jac;',jac_data.jac[0,...])
            jacobian = np.take(jac_data.jac[0,...],np.array(ov_idx_list),axis=-1)
            print('jacobian:',jacobian)
            print('jacobian shape:',jacobian.shape)
            
            #jacobian = np.ones((5384, 2, 5))
            #jac_data = Namespace(output_cols=['force','torque'])

            output_data_points = [o.shape[0] for o in self.getOutput(outdir)]
            ind = 0
            jacobians = []
            for n_points,device in zip(output_data_points,self.testing_devices):
                out_var = device.Dataframe_Colname_Output
                print(f'n_points {n_points},'
                    f'out_var_ind: {jac_data.output_cols.index(out_var)}')
                jacobians.append(jacobian[ind:ind+n_points,jac_data.output_cols.index(out_var),:])
                ind += n_points
            print('jacobians shape:',[j.shape for j in jacobians])
            return jacobians
        else:
            return None


    def getOutput(self,outdirs: List[Path],use_stress:bool=True) -> List[np.ndarray]:
        '''
        read the simulation output from outdir

        Args:
            outdir (:obj:'Path'): output directory of the simulation
        Returns:
            (list of :obj:'np.ndarray') each storing the output 
            produced for one simulation input file
        '''
        
        output_data = [device.get_response_data(use_stress=use_stress,geom=self.geometry,dirs=outdirs)
                for device in self.testing_devices]
        return output_data

    #@property
    #def has_grad(self)->bool:
    #    return True

    class SimJob(Simulation.SimJob):
        def __init__(self,sim:MetaModel,optVars: List[OptVars],throwExec: bool = True,
            del_outdir: bool = False, use_stress:bool = True, jacobian:bool=False) -> None:
            self.logger          = logging.getLogger(__name__)

            assert issubclass(type(sim),Simulation), "sim argument needs to be derived from Simulation"
            
            if isinstance(optVars,OptVars):
                optVars=[optVars]
        
            self.sim             = sim
            self.n_optvars       = len(optVars)
            self.optVars         = optVars
            self.throwExec       = throwExec
            self.del_outdir      = del_outdir

            self.outdirs         = None

            self._eval_callbacks = []
            self._use_stress = use_stress

            self.jacobian        = jacobian

        def start(self) ->None:
            '''
            should start the underlying calculation in a non-blocking way.
            Results will be returned using the blocking eval() method.
            
            we omitt this as the execution of the ML-Models is quite fast althoug parallel
            evaluation might be interesting in the future but we should do some timing before 
            '''
            pass

        def add_eval_callback(self,eval_callback):
            '''
            '''
            self._eval_callbacks.append(eval_callback)

        def eval(self) ->List[List[np.ndarray]]:
            '''
            Returns:
                list with an entry for each set of optimization variables that stores another list
                of :code:np.ndarray for each load mode
            '''
            if self.jacobian:
                assert len(self.optVars) == 1
            outdirs = None
            if self.sim.out_dir is not None:
                outdirs = [Path(tempfile.mkdtemp(
                        prefix=f'efi_cp_', dir=self.sim.out_dir))
                        for _ in range(len(self.optVars))]
            output = []
            for od,ov in zip(outdirs,self.optVars):
                if self.jacobian:
                    jac = self.sim.predict(ov,od,get_jac=True)
                    output.append(jac)
                else:
                    self.sim.predict(ov,od,get_jac=False)
                    output.append(self.sim.getOutput(od))
            #self.logger.debug(f'SimJob::_run() creating {outdir}')  

            #print('output: len(%d) lengths(%s)'%(len(output),[len(o) for o in output]))
            evalOutput=self._eval_impl(output)
            #print('evalOutput:len(%d) lengths(%s)'%(len(evalOutput),repr([len(o) for o in evalOutput])))

            self.outdirs = outdirs
            for cb in self._eval_callbacks:
                cb(evalOutput,outdirs)

            self.logger.debug('SimJob::eval() jobs finished')

            if self.jacobian:
                assert len(self.optVars) == 1
                evalOutput = evalOutput[0]

            return evalOutput

        def _eval_impl(self,output) ->List[np.ndarray]:
            '''
            can be overloaded by inheriting classes
            Args:
                output (list of :obj:'np.ndarray'): list containing the 
                    output of the simulation run
            Returns:
                unaltered output
            '''
            return output

    def get_input_disp_data(self,use_strain:bool = True):        
        input_data = [device.get_input_data(use_strain=use_strain,geom=self.geometry)
            for device in self.testing_devices]
        return input_data