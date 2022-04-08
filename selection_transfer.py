import numpy as np
import math
import pandas as pd
import os, sys
import random
from IPython import embed as shell

new_wd = "/Users/onderzoekl210/Desktop/Anne/masking-project"
old_wd = wd = "/Users/AnneZonneveld/Documents/STAGE/masking-project/"
practice = True

if practice == True:
    old_file = os.path.join(os.path.join(old_wd, 'help_files'), 'selection_THINGS_practice.csv')
else:
    old_file = os.path.join(os.path.join(old_wd, 'help_files'), 'selection_THINGS.csv')

old_df = pd.read_csv(old_file, sep=',', header=0)

columns = ['ImageID','concept','category', 'mask_type', 'mask_path']
new_df = pd.DataFrame(columns=columns)

lab = '2.10B'

for i in range(len(old_df)):

    print(f"i {i}")
    old_target_path = old_df.iloc[i]['ImageID'] 
    new_target_path = os.path.join(new_wd, '/'.join(old_target_path.split('/')[6:]))
    
    if old_df.iloc[i]['mask_path'] != 'no_mask':
        old_mask_path = old_df.iloc[i]['mask_path'] 
        new_mask_path = os.path.join(new_wd, '/'.join(old_mask_path.split('/')[6:]))
    else:
        new_mask_path = 'no_mask'

    row_info = [new_target_path, old_df['concept'].iloc[i], old_df['category'].iloc[i], old_df['mask_type'].iloc[i], new_mask_path]
    sub_df = pd.DataFrame([row_info], columns= columns)

    # Add to overall df
    new_df = pd.concat([new_df, sub_df])

# Export 
new_df = new_df.reset_index()
new_df = new_df.drop(columns=['index'])

if practice == True:
    new_df.to_csv(os.path.join(wd, 'help_files', f'selection_THINGS_practice_{lab}.csv')) 
else:   
    new_df.to_csv(os.path.join(wd, 'help_files', f'selection_THINGS_{lab}.csv')) 
