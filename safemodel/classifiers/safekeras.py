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
        
        print(f'args is a {type(args)} = {args}  kwargs is a {type(kwargs)}= {kwargs}')
        
        #initialise all the values that get provided as options to keras
        # and also l2 norm clipping and learning rates, batch sizes
        self.inputs = None
        if 'inputs' in kwargs.keys():
            print("inputs is in kwargs.keys")
            self.inputs=the_kwargs['inputs']
            print(self.inputs)
        elif len(args)==3: #defaults is for Model(input,outputs,names)
            print("running elif block")
            self.inputs= args[0]

        self.outputs= None
        if 'outputs' in kwargs.keys():
            self.outputs=the_kwargs['outputs']
        elif len(args)==3:
            self.outputs = args[1]

        if 'l2_norm_clip' in kwargs.keys():
            #set l2_norm_clip is supplied
            self.l2_norm_clip=the_kwargs['l2_norm_clip']
        else:
            #set l2_norm_clip to a default
            # preliminary_check(apply_constraints=True)
            # reads value in rules.json
            # value in rules.json will override this default  
            self.l2_norm_clip=1.0

        if 'noise_multiplier' in kwargs.keys():
            #set noise_multiplier if supplied
            self.noise_multiplier=the_kwargs['noise_multiplier']
        else:
            #set noise_multiplier to a default
            # preliminary_check(apply_constraints=True)
            # reads value in rules.json
            # value in rules.json will override this default  
            self.noise_multiplier = 0.5

        if 'min_epsilon' in kwargs.keys():
            #set noise_multiplier if supplied
            self.min_epsilon=the_kwargs['min_epsilon']
        else:
            #set to a default
            #preliminary_check(apply_constraints=True)
            # reads value in rules.json
            # value in rules.json will override this default  
            self.min_epsilon = 10

        if 'delta' in kwargs.keys():
            #set noise_multiplier if supplied
            self.delta=the_kwargs['delta']
        else:
            #set to a default
            # preliminary_check(apply_constraints=True)
            # reads value in rules.json
            # value in rules.json will override this default  
            self.delta = 1e-5
            
        if 'batch_size' in kwargs.keys():
            #set noise_multiplier if supplied
            self.delta=the_kwargs['batch_size']
        else:
            #set to a default
            # preliminary_check(apply_constraints=True)
            # reads value in rules.json
            # value in rules.json will override this default  
            self.batch_size = 25

        if 'num_microbatches' in kwargs.keys():
            #set noise_multiplier if supplied
            self.delta=the_kwargs['num_microbatches']
        else:
            #set to a default
            # preliminary_check(apply_constraints=True)
            # reads value in rules.json
            # value in rules.json will override this default  
            self.num_microbatches = None

        if 'num_microbatches' in kwargs.keys():
            #set noise_multiplier if supplied
            self.learning_rate=the_kwargs['learning_rate']
        else:
            #set to a default
            # preliminary_check(apply_constraints=True)
            # reads value in rules.json
            # value in rules.json will override this default  
            self.learning_rate = 0.1
            
        if 'optimizer' in kwargs.keys():
            self.optimizer = the_kwargs['optimizer']
        else:
            optimizer = tf_privacy.DPKerasSGDOptimizer

        KerasModel.__init__(self,inputs=self.inputs,outputs=self.outputs)
        #KerasModel.__init__(self)
        SafeModel.__init__(self)


        self.model_type: str = "KerasModel"
        super().preliminary_check(apply_constraints=True, verbose=True)
        #self.apply_specific_constraints()
        
        #need to move these two to rules.json so they can be set generally by TREs 
        #and read them in here
        
        #self.min_epsilon = 10 #get from json
        #self.delta = 1e-5  #get from json

        #optional- move this to json - not for nowe
        #self.batch_size=25
        #self.l2_norm_clip = 1.0 
        #self.noise_multiplier = 0.5
        self.num_microbatches = None
        #self.learning_rate = 0.1
        
    #need to be made available to user and provide better feedback if not true 
    #def dp_epsilon_met(num_examples=0:int,batch_size=0:int,epochs=0:int) ->bool:
    def dp_epsilon_met(self, num_examples=0,batch_size=0,epochs=0):
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

        if(self.optimizer == "None"):
            print("Changed parameter optimizer = 'DPKerasSGDOptimizer'")
            opt = tf_privacy.DPKerasSGDOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=learning_rate)
            
        if(self.optimizer == "Adagrad"):
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

        msg2, disclosive2 = check_optimizer_allowed(optimizer)
        if(disclosive2 == True):
            disclosive = True

        return msg+msg2, disclosive


            
    def posthoc_check(self, verbose: bool = True    ) -> tuple[str, bool]:
        """Checks whether model has been changed since fit() was last run and records eta"""

        msg, disclosive = super().posthoc_check()
        dpusedmessage, dpused = self.check_DP_used(self.optimizer)
        
        print(self.optimizer)
        allowedmessage, allowed = self.check_optimizer_allowed(self.optimizer)
        
        ##TODO call dp_epsilon_met()
        self.epochs = 20
        ok, current_epsilon = self.dp_epsilon_met(num_examples=self.num_samples, batch_size=self.batch_size, epochs=self.epochs)
        if(not ok):
            dpepsilonmessage = f"epsilon is not sufficient for Differential privacy: {current_epsilon}. You must modify one or more of batch_size, epochs, number of samples."
        else:
            dpepsilonmessage = f"epsilon is sufficient for Differential privacy: {current_epsilon}."

        theType= type(self.optimizer)
        print(f'optimiser is type {theType}')

        dpused,reason = self.check_DP_used(self.optimizer)
        msg2 = (f' It is {dpused} that the model will be DP because {reason}')

        msg = msg + msg2
        msg = msg + dpepsilonmessage
        print(msg)
        print(reason)
        return msg, reason
        
        #if that is ok and model has been fitted then still need to 
        
        
        

        # check if provided optimizer is one of the allowed types
        #dp_optimizers = (
        #    "tensorflow_privacy.DPKerasAdagradOptimizer",
        #    "tensorflow_privacy.DPKerasAdamOptimizer",
        #    "tensorflow_privacy.DPKerasSGDOptimizer",
        #)

class Safe_tf_DPModel(SafeModel, DPModel):
    """ Privacy Protected tensorflow_privacy DP-SGD subclass of Keras model"""

    def __init__(l2_norm_clip:float, noise_multiplier:float, use_xla:bool=True, *args:any, **kwargs:any) ->None:
        """creates model and applies constraints to parameters"""
        safemodel.__init__(self)
        DPModel.__init__(self, **kwargs)
        self.model_type: str = "tf_DPModel"
        super().preliminary_check(apply_constraints=True, verbose=True)



