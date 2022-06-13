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
        if 'l2_norm_clip' in kwargs.keys():
            self.l2_norm_clip=the_kwargs['l2_norm_clip']
        else: 
            self.l2_norm_clip=1.0

        if 'noise_multiplier' in kwargs.keys():
            self.noise_multiplier=the_kwargs['noise_multiplier']
        else: 
            self.noise_multiplier = 0.5

        if 'min_epsilon' in kwargs.keys():
            self.min_epsilon=the_kwargs['min_epsilon']
        else:
            self.min_epsilon = 10

        if 'delta' in kwargs.keys():
            self.delta=the_kwargs['delta']
        else:
            self.delta = 1e-5
            
        if 'batch_size' in kwargs.keys():
            self.batch_size = int(the_kwargs['batch_size'])
        else:
            self.batch_size = 25

        if 'num_microbatches' in kwargs.keys():
            self.num_microbatches=the_kwargs['num_microbatches']
        else: 
            self.num_microbatches = None

        if 'learning_rate' in kwargs.keys():
            self.learning_rate=the_kwargs['learning_rate']
        else:
            self.learning_rate = 0.1
            
        if 'optimizer' in kwargs.keys():
            self.optimizer = the_kwargs['optimizer']
        else:
            optimizer = tf_privacy.DPKerasSGDOptimizer

        if 'num_samples' i kwargs.keys():
            self.num_samples = the_kwargs['num_samples']
        else:
            num_samples = 0
            

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

        ok, current_epsilon = dp_epsilon_met(self,
                       num_examples=self.num_samples,
                       self.batch_size,
                       epochs=self.epochs)

        if ok:
            print(f"Current epsilon is {current_epsilon}")
            msg = "The requirements for DP are met, current epsilon is: {self.current_epsilon}. with the following parameters:  Num Samples = {self.num_samples}, batch_size = {self.batch_size}, epochs = {self.epochs}"
            return 0, current_epsilon, num_samples, batch_size, epochs
        if not ok:
            print(f"Current epsilon is {current_epsilon}")
            msg = f"The requirements for DP are not met, current epsilon is: {self.current_epsilon}. To attain true DP the following parameters can be changed:  Num Samples = {self.num_samples}, batch_size = {self.batch_size}, epochs = {self.epochs}"
            print(msg)
            return 1, current_epsilon, num_samples, batch_size, epochs
        
        
    def dp_epsilon_met(self, num_examples=0:int,batch_size=0:int,epochs=0:int) -> bool:
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

        batch_size=self.batch_size
        l2_norm_clip = self.l2_norm_clip 
        noise_multiplier = self.noise_multiplier
        num_microbatches = self.num_microbatches
        learning_rate = self.learning_rate
        
        #print(optimizer)
        if(self.optimizer == None):
            print("Changed parameter optimizer = 'DPKerasSGDOptimizer'")
            self.optimizer=tf_privacy.DPKerasSGDOptimizer
            opt = tf_privacy.DPKerasSGDOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=learning_rate)

        elif(self.optimizer == tf_privacy.DPKerasSGDOptimizer):
            opt = tf_privacy.DPKerasSGDOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=learning_rate)

        elif(self.optimizer == tf_privacy.DPKerasAdagradOptimizer):
            opt = tf_privacy.DPKerasAdagradOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=learning_rate)

        elif(self.optimizer == tf_privacy.DPKerasAdamOptimizer):
            opt = tf_privacy.DPKerasAdamOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=learning_rate)

        elif(self.optimizer == "None"):
            print("Changed parameter optimizer = 'DPKerasSGDOptimizer'")
            opt = tf_privacy.DPKerasSGDOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=learning_rate)
            
        elif(self.optimizer == "Adagrad"):
            print("WARNING: model parameters may present a disclosure risk")
            print("Changed parameter optimizer = 'DPKerasAdagradOptimizer'")
            opt = tf_privacy.DPKerasAdagradOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=learning_rate)
            
        elif(self.optimizer == "Adam"):
            print("WARNING: model parameters may present a disclosure risk")
            print("Changed parameter optimizer = 'DPKerasAdamOptimizer'")
            opt = tf_privacy.DPKerasAdamOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=learning_rate)
            
        elif(self.optimizer == "SGD"):
            print("WARNING: model parameters may present a disclosure risk")
            print("Changed parameter optimizer = 'DPKerasSGDOptimizer'")
            opt = tf_privacy.DPKerasSGDOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=learning_rate)
            
        else:
            print("WARNING: model parameters may present a disclosure risk")
            print(f"Unknown optimizer {self.optimizer} - Changed parameter optimizer = 'DPKerasSGDOptimizer'")
            opt = tf_privacy.DPKerasSGDOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=learning_rate)
            

        super().compile(opt, loss, metrics)
        ok, reason = self.check_DP_used(opt)
        print(f"DP optimizer used = {ok}")
        print(reason)

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
        


