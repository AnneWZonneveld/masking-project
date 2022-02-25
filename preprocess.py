"""
Pre processing for THINGS.
Creates a csv file containing following info for selected categories:
- ImageID: file path to particular image
- Concept: concept of particular image
- Category: category (Bottum-up) of particular image

by Anne Zonneveld, Febr 2022
"""

import numpy as np
from numpy.fft import fft2, ifft2
import pandas as pd
import os, sys
import random
from cmath import sqrt
from typing import cast
from PIL import Image
# from IPython import embed as shell

wd = '/Users/AnneZonneveld/Documents/STAGE/task/'

things_concept = pd.read_csv(os.path.join(wd, "help_files", "things_concepts.tsv"), sep='\t', header=0)
image_paths =  pd.read_csv(os.path.join(wd, "help_files", "image_paths.csv"), sep=',', header=None)

# categories = ["vegetable", "fruit", "drink", "insect", "bird", 
#             "clothing", "musical instrument", "body part", "plant", "sports equipment"] # use 5 

categories = ["vegetable", "drink", "insect", "clothing", "musical instrument"] 
mask_category = "body part" # masks should different category than target

masks = ['natural', 'scrambled', 'noise', 'geometric']

nr_concept = 5
nr_per_concept = 5
nr_mask_instances = 15
nr_repeats = 3

def scramble_images(images, p = 1, rescale = 'off'):
    """ Function to phase scramble image(s). 
        Based on the imscramble function by Martin Hebart for MATLAB.
        
        images: list with 2D (b&w) or 3D (colour) image arrays
        p = scrambling factor (default = 1)
        rescale = - 'off' (default)
                  - 'range' --> rescale to original range
        
        Saves images to mask_paths and returns list with all mask paths.
        """
    
    # for test purpose
    # images = []
    # for i in range(15):
    #     image = np.random.uniform(size = (imSize[0], imSize[1], 3))
    #     images.append(image)

    mask_dir = os.path.join(wd, 'stimuli', 'masks', 'scrambled')
    if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)    

    mask_paths = []

    for i in range(len(images)):

        image = images[i]
        imtype = type(image)
        #image = float(image)

        imSize = image.shape

        RandomPhase = p * np.angle(fft2(np.random.uniform(size = (imSize[0], imSize[1]))))
        RandomPhase[0] = 0 # leave out the DC value

        if len(imSize) == 2:
            imSize = (imSize[0], imSize[1] , 1)

        # preallocate
        imFourier = np.zeros(imSize, dtype = 'complex_')
        Amp = np.zeros(imSize, dtype = 'complex_')
        Phase = np.zeros(imSize, dtype = 'complex_')
        imScrambled = np.zeros(imSize, dtype = 'complex_')

        for layer in range(imSize[2]):
            imFourier[:, :, layer] = fft2(image[:,:,layer])
            Amp[:, :, layer] = abs(imFourier[:, :, layer])
            Phase[:, :, layer] = np.angle(imFourier[:, :, layer]) + RandomPhase

            # combine Amp and Phase for inverse Fourier
            imScrambled[:, :, layer] = ifft2(Amp[:,:,layer] * np.exp(sqrt(-1)) * (Phase[:,:,layer]))

        imScrambled = np.real(imScrambled)

        if rescale == 'range':
            minim = np.min(image)
            maxim = np.max(image)
            imScrambled =  minim + (maxim-minim) * np.divide((imScrambled - np.min(imScrambled)), (np.max(imScrambled) - np.min(imScrambled)))
        
        imScrambled = cast(imtype, imScrambled)

        im = Image.fromarray((imScrambled * 255).astype(np.uint8))
        im_name = os.path.join(mask_dir, 'scrambled_' + str(i) + '.jpg' ) 
        im.save(im_name)

        mask_paths.append(os.path.join(im_name.split('/')[-3], im_name.split('/')[-2], im_name.split('/')[-1]))

    return mask_paths

def noise_masks(self): 
    pass
    

def create_masks(self, things_concept = things_concept, image_paths = image_paths, mask_category=mask_category):
    """create / select masks"""

    all_masks = {'natural:' : [],
                 'scrambled' : [],
                 'noise' : [],
                 'geometric':[]}

    # Collect nr_mask_instance images from mask_category
    mask_concepts = things_concept['uniqueID'][things_concept['All Bottom-up Categories'].str.contains(mask_category)]
    mask_concepts = random.sample(mask_concepts.values.tolist(), nr_mask_instances)

    mask_dir = os.path.join(wd, 'stimuli', 'masks', 'natural')
    if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)   


    natural_mask_paths = []
    natural_mask_np_ims = []

    # Pick one image for every concept
    for concept in mask_concepts: 
        
        paths = image_paths[image_paths.iloc[:, 0].str.contains(concept)].values.tolist()

        # Correct for substrings
        corrected_paths = []
        for path in paths:
            if path[0].split('/')[1] == concept:
                corrected_paths.append(path[0])
        
        # Pick one 
        picked_path = random.choice(corrected_paths)

        # Convert image to array and add to list
        im_name = os.path.join(wd,'stimuli', picked_path)
        im = Image.open(im_name).convert('RGB')
        np_im = np.array(im)
        natural_mask_np_ims.append(np_im)

        # Add name to list 
        natural_mask_paths.append(picked_path)


    # natural mask
    all_masks['natural'] = natural_mask_paths

    # scrambled mask
    all_masks['scrambled'] = scramble_images(natural_mask_np_ims, rescale = 'range', p = 0.5) 
    
    # noise mask
    # function that creates nr_mask_instances noise masks

    # geometric mask
    # function that create nr_mask_instances geometric masks




def create_selection_csv(self, things_concept = things_concept, image_paths = image_paths, categories = categories):
    """creates csv that contains info for all trials:
    - imageID (path to target image)
    - concept
    - category
    - mask (path to mask)"""

    df = pd.DataFrame(columns= ['ImageID','concept','category', 'mask'])

    # Find according image paths for categories
    for category in categories:

        concepts = things_concept['uniqueID'][things_concept['All Bottom-up Categories'].str.contains(category)]

        # pick nr_concepts
        concepts = random.sample(concepts.values.tolist(), nr_concept)

        for concept in concepts: 
            
            paths = image_paths[image_paths.iloc[:, 0].str.contains(concept)].values.tolist()

            # Correct for substrings
            corrected_paths = []
            for path in paths:
                if path[0].split('/')[1] == concept:
                    corrected_paths.append(path[0])

            # pick nr_per_concept
            corrected_paths  = random.sample(corrected_paths, nr_per_concept)
                
            # Link masks
            for mask in masks:
                for i in range(nr_mask_instances):
                    sub_df = pd.DataFrame(columns= ['ImageID','concept','category', 'mask'])
                    sub_df['ImageID'] = np.array(corrected_paths)
                    sub_df['concept'] = np.repeat(concept, len(corrected_paths))
                    sub_df['category'] = np.repeat(category, len(corrected_paths))
                    sub_df['mask'] = np.repeat(f"{mask}_{i}", len(corrected_paths)) # change to mask path

                    for j in range(nr_repeats):
                        df = pd.concat([df, sub_df])

    # Export 
    df = df.reset_index()
    df = df.drop(columns=['index'])
    df.to_csv(os.path.join(wd, 'selection_THINGS.csv'))  
    

# Calculate prototypes per concept
# Calculate distance prototype and mask
