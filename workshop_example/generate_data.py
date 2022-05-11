'''Generate some toy data for the workshop'''

# %% Imports
from typing import List
import numpy as np
import pandas as pd

# %% Method to generate data
def make_toy_data(class_sizes: List[int],seed:int=1) -> pd.DataFrame:
    '''
    Make some toy data with health features
    Each feature gets a class-specific mean (or probability of 1/0)
    class_sizes is a list, with one entry for each class containing how many
    examples should be in the class
    '''
    np.random.seed(seed)
    output_df = None
    for i, class_size in enumerate(class_sizes):
        age_mean = np.random.uniform(10, 100)
        age_data = np.random.poisson(age_mean, class_size)
        smoker_prob = np.random.beta(0.1, 0.1)
        smoker_data = np.random.binomial(1, smoker_prob, class_size)
        employed_prob = np.random.beta(0.3, 0.3)
        employed_data = np.random.binomial(1, employed_prob, class_size)
        resting_hr_mean = np.random.uniform(40, 90)
        hr_data = np.random.poisson(resting_hr_mean, class_size)
        weight_mean = np.random.uniform(50, 110)
        weight_data = np.random.poisson(weight_mean, class_size)

        #jim trying to inject spedific eamples
        #if class_size==1:
        #    age_data=27
        #    smoker_data = 0
        #    employed_data= 0
        #    hr_data = 70
        #    weight_data= 85
            

        class_labels = i * np.ones(class_size, int)
        class_data = pd.DataFrame(
            {
                'age': age_data,
                'smoker': smoker_data,
                'resting_hr': hr_data,
                'weight': weight_data,
                'employed': employed_data,
                'label': class_labels
            }
        )
        if output_df is None:
            output_df = class_data
        else:
            output_df = pd.concat((output_df, class_data))

    return output_df


def main():
    output_df = make_toy_data([5, 10, 10])
    print(output_df)

if __name__ == '__main__':
    main()
