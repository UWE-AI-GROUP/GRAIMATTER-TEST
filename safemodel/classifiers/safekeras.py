import tensorflow as tf
import tensorflow_privacy as tf_privacy

from tensorflow.keras import Model as KerasModel
from safemodel.safemodel import SafeModel  
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras

import tensorflow_privacy as tf_privacy
from tensorflow_privacy import DPModel
from typing import Any
import sys



class Safe_KerasModel(KerasModel, SafeModel ):
    """Privacy Protected Wrapper around  tf.keras.Model class from tensorflow 2.8"""
    def dp_epsilon_met(self, num_examples:int, batch_size:int = 0 ,epochs:int = 0 ) -> bool:
        """Checks if epsilon is sufficient for Differential Privacy
           Provides feedback to user if epsilon is not sufficient"""
        privacy = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=num_examples,
                                              batch_size=batch_size,
                                              noise_multiplier=self.noise_multiplier,
                                              epochs=epochs,
                                              delta=self.delta)
        if privacy[0] < self.min_epsilon:
            ok= True
        else:
            ok= False
        return ok,privacy[0]
    

    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Creates model and applies constraints to params"""

        the_args = args
        the_kwargs = kwargs
                
        #initialise all the values that get provided as options to keras
        # and also l2 norm clipping and learning rates, batch sizes
        self.inputs = None
        if 'inputs' in kwargs.keys():
            self.inputs=the_kwargs['inputs']
        elif len(args)==3: #defaults is for Model(input,outputs,names)
            self.inputs= args[0]

        self.outputs= None
        if 'outputs' in kwargs.keys():
            self.outputs=the_kwargs['outputs']
        elif len(args)==3:
            self.outputs = args[1]

        # set values where the user has supplied them
        #if not supplied set to a value that preliminary_check
        # will over-ride with TRE-specific values from rules.json 

        defaults = {
            'l2_norm_clip':1.0,  
            'noise_multiplier': 0.5,
            'min_epsilon' : 10,
            'delta' : 1e-5,
            'batch_size' : 25,
            'num_microbatches' : None,
            'learning_rate': 0.1,
            'optimizer': tf_privacy.DPKerasSGDOptimizer,
            'num_samples': 250,
            'epochs': 20
                    }
        
        for key in defaults.keys():
            if key in kwargs.keys():
                setattr(self,key,kwargs[key])
            else:
                setattr(self,key,defaults[key])
                
        if(self.batch_size == 0):
            print("batch_size should not be 0 - division by zero")

        KerasModel.__init__(self,inputs=self.inputs,outputs=self.outputs)
        SafeModel.__init__(self)

        self.ignore_items = [
            '_jit_compile',
            'examine_seperately_items',
            'ignore_items',
            'loss',
            'researcher'
        ]

        self.model_type: str = "KerasModel"
        super().preliminary_check(apply_constraints=True, verbose=True)
        #self.apply_specific_constraints()

    def check_epsilon(self, num_samples:int,
                       batch_size:int,
                       epochs:int) -> bool:

        if (batch_size==0):
            print("Division by zero setting batch_size =1")
            batch_size=1 
            
        ok, current_epsilon = self.dp_epsilon_met(
                       num_examples=num_samples,
                       batch_size=batch_size,
                       epochs=epochs)

        if ok:
            print(f"Current epsilon is {current_epsilon}")
            msg = "The requirements for DP are met, current epsilon is: {current_epsilon}. with the following parameters:  Num Samples = {num_samples}, batch_size = {batch_size}, epochs = {epochs}"
            return True, current_epsilon, num_samples, batch_size, epochs
        if not ok:
            print(f"Current epsilon is {current_epsilon}")
            msg = f"The requirements for DP are not met, current epsilon is: {current_epsilon}. To attain true DP the following parameters can be changed:  Num Samples = {num_samples}, batch_size = {batch_size}, epochs = {epochs}"
            print(msg)
            return False, current_epsilon, num_samples, batch_size, epochs
        
        
    def fit(self,X,Y,validation_data, epochs, batch_size):
        ###TODO TIDY UP:
        print(X.shape)
        self.num_samples = X.shape[0]
        #make sure you are passing keywords through - but also checking batch size epochs
        ok, current_epsilon = self.dp_epsilon_met(X.shape[0], batch_size, epochs)
        
        if not ok:
            print(f"Current epsilon is {current_epsilon}")
            msg = f"The requirements for DP are not met, current epsilon is: {current_epsilon}. To attain true DP the following parameters can be changed:  Num Samples = {X.shape[0]}, batch_size = {batch_size}, epochs = {epochs}"
            print(msg)
            keepgoing = input('This will not result in a Differentially Private model do you want to continue [Y/N]')
            #behave appropriately
            if((keepgoing != 'Y') and (keepgoing != 'y') and (keepgoing !='yes') and (keepgoing != 'Yes')):
                print(f"Current epsilon is {current_epsilon}")
                print("Chose not to continue")
                
                sys.exit(0)
                
            else:
                print("Continuing")
        else:
            pass
        returnval = super().fit(X, Y, validation_data=validation_data, epochs=epochs, batch_size=batch_size)


        return returnval # super().fit
        
        
    def check_optimizer_is_DP(self, optimizer):
        DPused = False
        reason = "None"
        if ( "_was_dp_gradients_called" not in optimizer.__dict__ ):
            reason = "optimizer does not contain key _was_dp_gradients_called so is not DP."
            DPused = False
        else:
            reason = "optimizer does  contain key _was_dp_gradients_called so should be DP."
            DPused = True
        return DPused, reason

    def check_DP_used(self,optimizer):
        DPused = False
        reason = "None"
        if ( "_was_dp_gradients_called" not in optimizer.__dict__ ):
            reason = "optimizer does not contain key _was_dp_gradients_called so is not DP."
            DPused = False
        elif (optimizer._was_dp_gradients_called==False):
            reason= "optimizer has been changed but fit() has not been rerun."
            DPused = False
        elif (optimizer._was_dp_gradients_called==True):
            reason= " value of optimizer._was_dp_gradients_called is True so DP variant of optimizer has been run"
            DPused=True
        else:
            reason = "unrecognised combination"
            DPused = False
            
        return DPused, reason

    def check_optimizer_allowed(self, optimizer):
        disclosive = True
        reason = "None"
        allowed_optimizers = [
            "tensorflow_privacy.DPKerasAdagradOptimizer",
            "tensorflow_privacy.DPKerasAdamOptimizer",
            "tensorflow_privacy.DPKerasSGDOptimizer"
        ]
        print(f"{str(self.optimizer)}")
        if(self.optimizer in allowed_optimizers):
            discolsive = False
            reason = f"optimizer {self.optimizer} allowed"  
        else:
            disclosive = True
            reason = f"optimizer {self.optimizer} is not allowed"

        return disclosive, reason
    

    
    def compile(self,optimizer=None, loss='categorical_crossentropy', metrics=['accuracy']):

        replace_message= "WARNING: model parameters may present a disclosure risk"
        using_DP_SGD= "Changed parameter optimizer = 'DPKerasSGDOptimizer'"
        Using_DP_Adagrad= "Changed parameter optimizer = 'DPKerasAdagradOptimizer'"
        using_DP_Adam="Changed parameter optimizer = 'DPKerasAdamOptimizer'"
        
        optimizer_dict={
            None:(using_DP_SGD,tf_privacy.DPKerasSGDOptimizer),
            tf_privacy.DPKerasSGDOptimizer:("", tf_privacy.DPKerasSGDOptimizer),
            tf_privacy.DPKerasAdagradOptimizer: ("", tf_privacy.DPKerasAdagradOptimizer),
            tf_privacy.DPKerasAdamOptimizer:("",tf_privacy.DPKerasAdamOptimizer),
            "Adagrad" : (replace_message+Using_DP_Adagrad, tf_privacy.DPKerasAdagradOptimizer),
            "Adam": (replace_message+using_DP_Adam, tf_privacy.DPKerasAdamOptimizer),
            "SGD": (replace_message+using_DP_SGD, tf_privacy.DPKerasSGDOptimizer)
        }
        
        val= optimizer_dict.get(self.optimizer,"unknown")
        if val=="unknown":
            opt_msg = using_DP_SGD
            opt_used= tf_privacy.DPKerasSGDOptimizer
        else:
            opt_msg= val[0]
            opt_used = val[1]
            
        opt= opt_used ( l2_norm_clip=self.l2_norm_clip,
                        noise_multiplier=self.noise_multiplier,
                        num_microbatches=self.num_microbatches,
                        learning_rate=self.learning_rate)
        
        if len(opt_msg)>0:
            print(opt_msg)


        super().compile(opt, loss, metrics)

    def additional_checks(
            self, curr_separate: dict, saved_separate: dict  ) -> tuple[str, str]:
        """Placeholder function for additional posthoc checks e.g. keras this
        version just checks that any lists have the same contents"""
        # posthoc checking makes sure that the two dicts have the same set of
        # keys as defined in the list self.examine_separately
        msg = ""
        disclosive = False
        for item in self.examine_seperately_items:
            if isinstance(curr_separate[item], list):
                if len(curr_separate[item]) != len(saved_separate[item]):
                    msg += f"Warning: different counts of values for parameter {item}"
                    disclosive = True
                else:
                    for i in range(len(saved_separate[item])):
                        difference = list(
                            diff(curr_separate[item][i], saved_separate[item][i])
                        )
                        if len(difference) > 0:
                            msg += (
                                f"Warning: at least one non-matching value"
                                f"for parameter list {item}"
                            )
                            disclosive = True
                            break


        ok, reason = check_optimizer_is_DP( self.optimizer )
        if(ok):
          msg2 = "- DP - Differentially private optimizer has been used"
        else:
          disclosive = True
          msg2 = "- Not DP -Standard (disclosive) optimizer has been used"

          msg = msg + msg2
          return msg, disclosive

        return msg+msg2, disclosive


            
    def posthoc_check(self, verbose: bool = True    ) -> tuple[str, bool]:
        """Checks whether model has been changed since fit() was last run and records eta"""

        msg, disclosive = super().posthoc_check()
        dpusedmessage, dpused = self.check_DP_used(self.optimizer)
        
        print(self.optimizer)
        allowedmessage, allowed = self.check_optimizer_allowed(self.optimizer)
        
        #call dp_epsilon_met()

        ok, current_epsilon = self.dp_epsilon_met(num_examples=self.num_samples, batch_size=self.batch_size, epochs=self.epochs)
        if(not ok):
            dpepsilonmessage = f"; however, epsilon is not sufficient for Differential privacy: {current_epsilon}. You must modify one or more of batch_size, epochs, number of samples."
        else:
            dpepsilonmessage = f" and epsilon is sufficient for Differential privacy: {current_epsilon}."

        dpused,reason = self.check_DP_used(self.optimizer)
        if(dpused):
            msg2 = (f' The model will be DP because {reason}')
        else:
            msg2 = (f' The model will not be DP because {reason}')

        msg = msg + msg2
        msg = msg + dpepsilonmessage
        
        return msg, reason
        


