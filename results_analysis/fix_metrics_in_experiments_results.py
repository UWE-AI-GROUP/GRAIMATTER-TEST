import os
import numpy as np
import pandas as pd

path = "/home/ec2-user/studies/GRAIMatter/experiments"

file_names = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
print(file_names)

for f in file_names:
    print(f)
    results_df = pd.read_csv(os.path.join(path, f))
    #this chunk is when a problem was found with the metrics after running the experiments
    tmp_min = results_df.mia_FMAX#wrong label, FMAX is in reality FMIN
    results_df['mia_FMIN']=[round(x,8) for x in results_df.mia_FMAX]
    results_df.mia_FMAX = [round(x,8) for x in tmp_min+results_df.mia_FDIF]
    results_df.mia_FDIF = [round(x,8) for x in results_df.mia_FDIF]
    results_df.replace([np.inf, -np.inf], 100.0, inplace=True)
    results_df.to_csv(os.path.join(path, f.split('.')[0]+'_corrected.csv'))