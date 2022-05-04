import tensorflow as tf

#import tensorflow.keras.Model as KerasModel
import safemodel
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras

import tensorflow_privacy as tf_privacy
from tensorflow_privacy import DPModel

class Safe_KerasModel(tf.keras.Model, safemodel):
    """Privacy Protected Wrapper around  tf.keras.Model class from tensorflow 2.8"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Creates model and applies constraints to params"""
        safemodel.__init__(self)
        KerasModel.__init__(self, *args, **kwargs)
        self.model_type: str = "KerasModel"
        super().apply_constraints(**kwargs)
        self.apply_specific_constraints()

    def check_DP_used(optimizer):
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

class Safe_tf_DPModel(safemodel, DPModel):
    """ Privacy Protected tensorflow_privacy DP-SGD subclass of Keras model"""

    def __init__(l2_norm_clip:float, noise_multiplier:float, use_xla:Bool=True, *args:any, **kwargs:any) ->None:
        """creates model and applies constraints to parameters"""
        safemodel.__init__(self)
        DPModel.__init__(self, **kwargs)
        self.model_type: str = "tf_DPModel"
        super().preliminary_check(apply_constraints=True, verbose=True)



