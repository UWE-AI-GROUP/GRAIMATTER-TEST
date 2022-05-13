import tensorflow as tf

from tensorflow.keras import Model as KerasModel
from safemodel.safemodel import SafeModel  
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras

import tensorflow_privacy as tf_privacy
from tensorflow_privacy import DPModel
from typing import Any


class Safe_KerasModel(KerasModel, SafeModel ):
    """Privacy Protected Wrapper around  tf.keras.Model class from tensorflow 2.8"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Creates model and applies constraints to params"""
        the_args = args
        the_kwargs = kwargs
        print(f'args is a {type(args)} = {args}  kwargs is a {type(kwargs)}= {kwargs}')
        #initialise all the values that get provided as options ot keras
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
            self.outputs = inputs[1]
        KerasModel.__init__(self,inputs=self.inputs,outputs=self.outputs)
        #KerasModel.__init__(self)
        SafeModel.__init__(self)


        self.model_type: str = "KerasModel"
        super().preliminary_check(apply_constraints=True, verbose=True)
        #self.apply_specific_constraints()

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def check_DP_used(self,optimizer):
        DPused = False
        reason = "None"
        if ( "_was_dp_gradients_called" not in optimizer.__dict__ ):
            reason = "optimiser does not contain key _was_dp_gradients_called so is not DP."
            DPused = False
        elif (optimizer._was_dp_gradients_called==False):
            reason= "optimiser has been changed but fit() has not been rerun."
            DPused = False
        elif (optimizer._was_dp_gradients_called==True):
            reason= " value of optimizer._was_dp_gradients_called is True so DP variant of optimiser has been run"
            DPused=True
        else:
            reason = "unrecognised combination"
            DPused = False
            
        return DPused, reason

    def compile(self):
        import tensorflow_privacy as tf_privacy
        batch_size=1
        l2_norm_clip = 1.5
        noise_multiplier = 1.3
        num_microbatches = batch_size
        learning_rate = 0.25

        if(self.optimizer == "None"):
            opt = tf_privacy.DPKerasSGDOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate)

        if(self.optimizer == "Adagrad"):
            opt = tf_privacy.DPKerasAdagradOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=learning_rate)

        elif(self.optimizer == "Adam"):
            opt = tf_privacy.DPKerasAdamOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=learning_rate)

        elif(self.optimizer == "SGD"):
            opt = tf_privacy.DPKerasSGDOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate)

        else:
            opt = tf_privacy.DPKerasSGDOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate)


        super().compile(opt)

    def check_optimizer_allowed(optimizer):
        disclosive = True
        reason = "None"
        allowed_optimizers = [
            "tensorflow_privacy.DPKerasAdagradOptimizer",
            "tensorflow_privacy.DPKerasAdamOptimizer",
            "tensorflow_privacy.DPKerasSGDOptimizer"
        ]
        if(optimizer in allowed_optimizers):
            discolsive = False
            reason = f"optimizer {optimizer} allowed"  
        else:
            disclosive = True
            reason = f"optimizer {optimizer} is not allowed"

        return reason, disclosive
    
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


            
    def posthoc_check(
        self, verbose: bool = True    ) -> tuple[str, bool]:
        """Checks whether model has been changed since fit() was last run and records eta"""

        msg, disclosive = super.posthoc(self,verbose,apply_constraints)
        dpusedmessage, dpused = self.check_DP_used(self.optimizer)
        
        print(optimizer)
        allowedmessage, allowed = self.check_optimizer_allowed(optimizer)

        return allowedmsg, reason
        
        #if that is ok and model has been fitted then still need to 
        
        
        

        # check if provided optimiswer is one of the allowed types
        #dp_optimisers = (
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



