"""
Pre processing for THINGS.
Creates a csv file containing following info for selected categories:
- ImageID: file path to particular image
- Concept: concept of particular image
- Category: category (Bottum-up) of particular image

by Anne Zonneveld, Febr 2022
"""

import numpy as np
import math
from numpy.fft import fft2, ifft2, fftn, ifftn 
import pandas as pd
import os, sys
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from cmath import sqrt
from typing import cast
from PIL import Image
from itertools import permutations 
from decimal import Decimal    
import torch
import torchvision.transforms as T         
# import colorednoise as cn
# from IPython import embed as shell

wd = '/Users/AnneZonneveld/Documents/STAGE/masking-project/'
#wd = '/home/c11645571/masking-project'

things_concept = pd.read_csv(os.path.join(wd, "help_files", "things_concepts.tsv"), sep='\t', header=0)
image_paths =  pd.read_csv(os.path.join(wd, "help_files", "image_paths.csv"), sep=',', header=None)

target_category = ["vegetable", "insect", "clothing", "musical instrument"] 
mask_category = ["bird", "furniture"] # masks should different category than target

masks = ['1_natural', '2_scrambled', '3_noise', '4_geometric', '5_lines', '6_blocked']


nr_concepts = 10
nr_per_concept = 4
nr_mask_instances = 20
nr_repeats = 3

def crop_image(image, root, image_dimensions=(480, 480)):
    """
    Function to center crop images. Saves cropped images to file path
    
    - images: list of tuples ---> (PIL image, image_name)
    - image_dimensions: dimensions to which the image has to be cropped
    """

    # check if expirment or DNN_analysis 
    
    if root.split('/')[-2] == 'DNN_analysis':
        cropped_dir = os.path.join(root, '1_natural')
        if not os.path.exists(cropped_dir):
                os.makedirs(cropped_dir)  
    else:
        cropped_dir = root

    center_crop = T.CenterCrop(size=image_dimensions)(image[0])

    # save created mask
    im_name = os.path.join(cropped_dir, image[1]) 
    center_crop.save(im_name)

    return im_name

def scramble_images_v1(images, root, p = 0.5, rescale = 'off'):
    """ Function to phase scramble image(s). 
        Based on the imscramble function by Martin Hebart for MATLAB.
        
        images: list with 2D (b&w) or 3D (colour) image arrays
        p = scrambling factor (default = 1)
        rescale = - 'off' (default)
                  - 'range' --> rescale to original range
        
        Saves images to mask_paths and returns list with all mask paths.
        """
    
    mask_dir = os.path.join(root, '2_scrambled')
    if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)    

    mask_paths = []

    for i in range(len(images)):

        image = images[i]/255
        imtype = type(image)

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
            RandomPhase = p * np.angle(fft2(np.random.uniform(size = (imSize[0], imSize[1]))))
            RandomPhase[0] = 0 

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
        im_name = os.path.join(mask_dir, 'v1_scrambled_' + str(i) + f'_p{p}.png' ) 
        im.save(im_name)

        mask_paths.append(os.path.join(im_name.split('/')[-3], im_name.split('/')[-2], im_name.split('/')[-1]))

    return mask_paths

def scramble_images_v2(images, root,  p = 0.5, rescale = 'off'):
    """ Function to phase scramble image(s). 
        Based on the imscramble function by Martin Hebart for MATLAB.
        BUT: does not add extra phase, but randomly swaps phase of existing images.
        
        images: list with 2D (b&w) or 3D (colour) image arrays
        p: scrambling factor (default = 1)
        rescale:  - 'off' (default)
                  - 'range' --> rescale to original range
        
        Saves images to mask_paths and returns list with all mask paths.
        """
    
    mask_dir = os.path.join(root, '2_scrambled')
    if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)    

    mask_paths = []

    for i in range(len(images)):

        # rescale to 0-1 values
        image = images[i]/255
        
        imtype = type(image)
        imSize = image.shape

        # RandomPhase = p * np.angle(fft2(np.random.uniform(size = (imSize[0], imSize[1]))))
        # RandomPhase[0] = 0 # leave out the DC value

        if len(imSize) == 2:
            imSize = (imSize[0], imSize[1] , 1)

        # preallocate
        imFourier = np.zeros(imSize, dtype = 'complex_')
        Amp = np.zeros(imSize, dtype = 'complex_')
        Phase = np.zeros(imSize, dtype = 'complex_')
        imScrambled = np.zeros(imSize, dtype = 'complex_')

        for layer in range(imSize[2]):
            print(f"layer : {layer}")
            imFourier[:, :, layer] = fft2(image[:,:,layer])
            Amp[:, :, layer] = abs(imFourier[:, :, layer])

            phase = np.angle(imFourier[:, :, layer])

            # perform x percent of permutations based on scrambling factor
            nr_permutations = math.perm(phase.size, 1)
            perms = int(p * nr_permutations)
            
            # permute 
            for perm in range(perms):
                print(f"{perm}/{perms}")
                i1, i2, i3, i4 = random.sample(range(0, image.shape[0]), 4)
                phase[i1, i2], phase[i3, i4] = phase[i3, i4], phase[i1, i2]

            Phase[:, :, layer] = phase

            # combine Amp and Phase for inverse Fourier
            imScrambled[:, :, layer] = ifft2(Amp[:,:,layer] * np.exp(sqrt(-1)) * (Phase[:,:,layer]))

        imScrambled = np.real(imScrambled)

        if rescale == 'range':
            minim = np.min(image)
            maxim = np.max(image)
            imScrambled =  minim + (maxim-minim) * np.divide((imScrambled - np.min(imScrambled)), (np.max(imScrambled) - np.min(imScrambled)))
        
        imScrambled = cast(imtype, imScrambled)

        im = Image.fromarray((imScrambled * 255).astype(np.uint8))
        im_name = os.path.join(mask_dir, 'v2_scrambled_' + str(i) + f'_p{p}.png' ) 
        im.save(im_name)

        mask_paths.append(os.path.join(im_name.split('/')[-3], im_name.split('/')[-2], im_name.split('/')[-1]))

    return mask_paths


def block_scrambled(n_masks, root, target_size = (480, 480), block_size=(120,120)):
    """
    Function to create block scrambled masks based on natural images. 
    """

    mask_dir = os.path.join(root, '6_blocked')
    if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)    

    img_dir = os.path.join(root, '1_natural')
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    num_blocks = [int(np.ceil(target_size[0] / block_size[0])), int(np.ceil(target_size[1] /block_size[1]))]

    block_mask_paths = []
    df_rows = []
   
    for m in range(n_masks):

        print(f'creating block mask {m}')
       
        # matrix of target size
        block_mask = np.zeros((target_size[0], target_size[1], 3))

        # set up to store relevant info
        image_paths = []
        if m == 0:
            col_names = []

        # loop through blocks
        for x in range(num_blocks[0]):
            for y in range(num_blocks[1]):

                # pick a random image
                pick = img_files[np.random.randint(len(img_files))]
                img = Image.open(os.path.join(img_dir , pick))
                
                # save which image picked from

                img = np.asarray(img, dtype='float64')

                # pick random block of image
                block_offset = [np.random.randint(img.shape[0]-block_size[0]), np.random.randint(img.shape[1]-block_size[1])]
                img_block = img[block_offset[0]:block_offset[0]+block_size[0], block_offset[1]:block_offset[1]+block_size[1],:]

                # fill mask with random block
                block_mask[x*block_size[0]:(x+1)*block_size[0],y*block_size[1]:(y+1)*block_size[1],:] = img_block

                # save info
                if m == 0:  
                    col_names.append(f'{x}_{y}')
                image_paths.append(pick)

        # save im
        mask_im = Image.fromarray((block_mask).astype(np.uint8))
        mask_name = os.path.join(mask_dir, f'blocked_{m}_b{num_blocks[0]}.png')

        mask_im.save(mask_name)
        block_mask_paths.append(mask_name)

        # save info        
        row_info = [f'blocked_{m}_b{num_blocks[0]}.png']
        row_info = row_info + image_paths
        df_rows.append(row_info)
    
    # create final ddf
    df = pd.concat([pd.DataFrame([i], columns= ['mask_name'] + col_names) for i in df_rows], ignore_index=True)
    df_path = os.path.join(wd, 'help_files', 'blocked_masks', root.split('/')[-2])   
    if not os.path.exists(df_path):
        os.makedirs(df_path)

    df.to_csv(os.path.join(df_path, f'block_masks_b{num_blocks}.csv'))
    
    return block_mask_paths


def noise_masks(n, root, image_dimensions=(480, 480), beta=2):
    """Function to create 3D noise mask.
    sz_x: size of mask in x-dimension
    sz_y: size of mask in y-dimension
    beta: PSD (power spectrum density) spectral slope
    """

    mask_dir = os.path.join(root, '3_noise')

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    mean = 0
    sigma = 1

    random_mats = np.random.normal(mean, sigma, (n,) + image_dimensions + (3,))

    # Normalize to 0-1
    random_mats = (random_mats / 4.96 + 1) * 0.5

    # 1/f mask

    x = np.linspace(-image_dimensions[0] // 2, image_dimensions[0] // 2, image_dimensions[0])
    y = np.linspace(-image_dimensions[0] // 2, image_dimensions[0] // 2, image_dimensions[0])
    xx, yy = np.meshgrid(x, y)

    mask = 1/np.sqrt(abs(xx)**2 + abs(yy)**2)**beta

    mask = mask[:,:, np.newaxis] * np.ones((1,1,3))
 
    # help tool
    # plot_mask = mask.copy()


    # plot_mask = plot_mask - plot_mask.min()
    # plot_mask = plot_mask / plot_mask.max()

    # plot_mask_im = Image.fromarray((plot_mask * 255).astype(np.uint8))
    # plot_mask_name = os.path.join(mask_dir, 'plot_mask' + f'_b{beta}.png')
    # plot_mask_im.save(plot_mask_name)

    mask_paths = []

    for m in range(n):
        fshift = np.fft.fftshift(np.fft.fftn(random_mats[m]))
        f_ishift = np.fft.ifftshift(fshift * mask)
        img_back = np.fft.ifftn(f_ishift)
        img_back = np.real(img_back)

        # normalize to 0-1
        img_back = img_back - img_back.min()
        img_back = img_back / img_back.max()

        # save created mask
        im = Image.fromarray((img_back * 255).astype(np.uint8))
        im_name = os.path.join(mask_dir, 'noise_' + str(m) + f'_b{float(beta)}.png')
        im.save(im_name)

        mask_paths.append(os.path.join(im_name.split('/')[-3], im_name.split('/')[-2], im_name.split('/')[-1]))

    return mask_paths


def geometric_masks(n_masks, root, sz_x = 480, sz_y = 480, d = 8):
        """ Function to create mondriaan masks. 
        Based on the make_mondrian_mask function by Martin Hebart for MATLAB.
        Creates square colored mondriaan masks.

        sz_x: size of mask in x-dimension
        sz_y: size of maks in y-dimension
        n_masks: number of masks to be created
        d: density, nr of shapes in image (default = 8)
        
        Saves images to mask_paths and returns list with all mask paths.
        """

        mask_dir = os.path.join(root, '4_geometric')
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)    

        mask_paths = []

        # possbile square sizes (in percent of x-dimension)
        sizes = np.arange(0.04, 0.18, 0.01)
        sizes = sizes *1.2

        # how many shapes should be drawn?
        # if area filled 8 times, normally sufficient according to hebbart
        # loop_nr = int(np.round(density*(sz_y/sz_x)/np.mean(np.exp(sizes)))) 
        loop_nr = d

        colors  =  np.array(
            [[1, 0 , 0],     # red
            [0.5, 0, 0],     # dark red
            [0, 1, 0],       # green
            [0, 0.5, 0],     # dark green
            [0, 0, 1],       # blue
            [0, 0 , 0.5],    # dark blue
            [0.1, 0.6, 1],   # light blue 
            [1, 1, 0],       # yellow
            [0.5, 0.5, 0],   # dark yellow
            [1, 0, 1],       # magenta
            [0, 1, 1],       # cyan
            [0, 0.5, 0.5],   # dark cyan
            [1, 1, 1],       # white
            [0, 0, 0],       # black
            #[0.5, 0.5, 0.5], # gray
            [0.3, 0.1, 0.5],  # dark purple
            [1, 0.7, 0]]      # orange
        )

        masks = {}
        sizes = np.ceil(sizes * sz_x).astype(int)  # sizes relative to x-dimension

        # make it two level
        mask_3d = 0.5 * np.ones((int(sz_y + np.max(sizes)), int(sz_x + np.max(sizes)), 3)) # background start colour is gray
        mask = mask_3d[:,:,1] 
        mask_index = np.arange(mask.size).reshape((mask.shape[0], mask.shape[1]))

        sizes_templates = {}

        # create squares
        for i in range(len(sizes)):
            template = 0 * mask
            square = np.ones((int(sizes[i]), int(sizes[i])))
            template[0:square.shape[1], 0:square.shape[0]] = square
            sizes_templates.update({f"{sizes[i]}" : template})

        # excluded = mask_index[mask_index.shape[0] - int(np.max(sizes)): mask_index.shape[0] + 1, mask_index.shape[1] - int(np.max(sizes)): mask_index.shape[1] - 1]
        excluded_y = mask_index[:, mask_index.shape[1] - int(np.max(sizes)) - 1 : mask_index.shape[1]]
        excluded_x = mask_index[mask_index.shape[0] - int(np.max(sizes)) -1 : mask_index.shape[0], :]
        excluded = np.append(excluded_y, excluded_x)

        # help
        # mask_3d[:, mask_index.shape[1] - int(np.max(sizes)) - 1 : mask_index.shape[1], :] = 0
        # mask_3d[mask_index.shape[0] - int(np.max(sizes)) -1 : mask_index.shape[0], :, :] = 0
        # im = Image.fromarray((mask_3d* 255).astype(np.uint8))
        # im_name = os.path.join(mask_dir, 'help' + '.png' ) 
        # im.save(im_name)

        mask_index = mask_index.flatten()
        mask_index = np.setdiff1d(mask_index, excluded.flatten())

        for i_mask in range(n_masks):
            
            # randomize maskindex for later starting position
            curr_mask_index = []

            for i in range(loop_nr):
                random_index = np.random.choice(mask_index)
                # random_index = np.where(mask_index == random_index)[0]
                curr_mask_index.append(random_index)

            created_mask = mask_3d.copy()

            for i_loop in range(loop_nr):

                #csize = int(np.ceil(len(sizes) * np.random.random_sample())) # random current size
                csize = int(np.random.choice(sizes))

                randpos = curr_mask_index[i_loop] 
                randpos_index = np.where(mask_index == randpos)[0]

                #currindex  = sizes_templates[f"{csize}"].astype(int).flatten() + randpos
                currindex = np.where(sizes_templates[f"{csize}"].astype(int).flatten() == 1)[0] + randpos_index
                currlevel = colors[np.random.randint(colors.shape[0]), :]
                
                R = created_mask[:,:,0].flatten()
                G = created_mask[:,:,1].flatten()
                B = created_mask[:,:,2].flatten()

                R[currindex] = currlevel[0]
                G[currindex] = currlevel[1]
                B[currindex] = currlevel[2]

                R = R.reshape(mask.shape[0], mask.shape[1])
                G = G.reshape(mask.shape[0], mask.shape[1])
                B = B.reshape(mask.shape[0], mask.shape[1])

                created_mask = np.dstack((R,G,B))
            
            # cropped_mask = created_mask[np.ceil(np.max(sizes)/2).astype(int): np.ceil(np.max(sizes)/2).astype(int) + sz_y, np.ceil(np.max(sizes)/2).astype(int): np.ceil(np.max(sizes)/2).astype(int) + sz_x, :]
            cropped_mask = created_mask[0:sz_y, np.ceil(np.max(sizes)/2).astype(int): np.ceil(np.max(sizes)/2).astype(int) + sz_x, :]
            masks.update({f"{i_mask}" : cropped_mask})

            # save created mask
            im = Image.fromarray((cropped_mask* 255).astype(np.uint8))
            im_name = os.path.join(mask_dir, 'geometric_' + str(i_mask) + f'_d{d}.png' ) 
            im.save(im_name)

            mask_paths.append(os.path.join(im_name.split('/')[-3], im_name.split('/')[-2], im_name.split('/')[-1]))
        
        return mask_paths

def line_masks(n_masks, root, sz_x = 480, sz_y = 480, d = 200):
        """ Function to create line masks. 

        sz_x: size of mask in x-dimension
        sz_y: size of maks in y-dimension
        n_masks: number of masks to be created
        d: density, nr of shapes in image (default = 8)
        
        Saves images to mask_paths and returns list with all mask paths.
        """

        mask_dir = os.path.join(root, '5_lines')
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)    

        mask_paths = []

        colors = ['red',
        'darkred',
        'green',
        'darkgreen',
        'blue',
        'darkblue',
        'lightblue',
        'yellow',
        'magenta',
        'cyan',
        'darkcyan',
        'white',
        'black',
        'mediumpurple',
        'orange'
        ]


        # create background with extra padding 
        padding = 10
        pad_x = sz_y + padding
        pad_y = sz_x + padding
        background = 0.5 * np.ones((pad_x, pad_y, 3)) # background gray
        background_im = Image.fromarray((background* 255).astype(np.uint8))

        # set line values
        lw = 5
        length = np.max(np.array([sz_y + padding, sz_x + padding]))

        # loop through n_masks
        for m in range(n_masks):

            # create fig
            fig, ax = plt.subplots()
            ax.imshow(background_im)

            # loop through all lines
            for line in range(d):
                anchor = (random.sample(range(pad_x), 1)[0], random.sample(range(pad_y), 1)[0])
                color = random.choice(colors)
                angle = random.sample(range(360), 1)[0]
                ax.add_patch(Rectangle(anchor, lw, length, angle=angle, 
                facecolor = color,
                fill=True,))

            # save
            im_name = os.path.join(mask_dir, 'lines_' + str(m) + f'_d{d}.png' ) 
            ax.axis('off')
            fig.savefig(im_name, bbox_inches='tight', pad_inches = 0)

            # crop 
            im = Image.open(im_name).convert('RGB')
            np_im = np.array(im)
            im.thumbnail(size=(sz_x, sz_y))
            im_name = os.path.join(mask_dir, 'lines_' + str(m) + f'_d{d}.png' ) 
            im.save(im_name, optimize=True, quality=100)

            mask_paths.append(os.path.join(im_name.split('/')[-3], im_name.split('/')[-2], im_name.split('/')[-1]))
            plt.clf()

            # # check
            # im = Image.open(im_name).convert('RGB')
            # np_im = np.array(im)
            print(f"creating line mask {m}")
        
        return mask_paths

# def create_targets(paths):
#     experiment_dir = os.path.join(wd, 'stimuli', 'experiment')
#     image_dir = os.path.join(experiment_dir, 'images')

#     if not os.path.exists(image_dir):
#         os.makedirs(image_dir)   

#     for path in paths:

#         image_name = path[7:]
 
#         # Get image from image base and convert 
#         im_path = os.path.join(wd, 'image_base', path[7:])
#         im = Image.open(im_path).convert('RGB')

#         # Crop
#         cropped_path = crop_image(image = (im, image_name), root = image_dir)


#     pass
  

def create_masks(things_concept = things_concept, image_paths = image_paths, mask_category=mask_category, type = 'experiment'):
    """create / select masks"""

    all_masks = {'natural' : [],
                 'scrambled' : [],
                 'noise' : [],
                 'geometric':[],
                 'lines': [],
                 'block':[]}

    if type == 'DNN_analysis': 

        # Collect nr_mask_instance images from mask_category
        mask_concepts = things_concept['uniqueID'][things_concept['All Bottom-up Categories'].str.contains(mask_category)]
        mask_concepts = random.sample(mask_concepts.values.tolist(), nr_mask_instances)

        DNN_analysis_dir = os.path.join(wd, 'stimuli', 'DNN_analysis')
        mask_dir = os.path.join(DNN_analysis_dir, 'masks')
        image_dir = os.path.join(DNN_analysis_dir, 'images')

        nat_mask_dir = os.path.join(mask_dir, '1_natural')
        if not os.path.exists(nat_mask_dir):
            os.makedirs(nat_mask_dir)   

        cropped_paths = []
        cropped_mask_np_ims = []


        # # Pick one image for every concept
        # for concept in mask_concepts: 

        #     paths = image_paths[image_paths.iloc[:, 0].str.contains(concept)].values.tolist()

        #     # Correct for substrings
        #     corrected_paths = []
        #     for path in paths:
        #         if path[0].split('/')[1] == concept:
        #             corrected_paths.append(path[0])

        #     # Pick one 
        #     picked_path = random.choice(corrected_paths)

        #     # Convert image to array and add to list
        #     im_name = os.path.join(wd,'stimuli', picked_path)
        #     im = Image.open(im_name).convert('RGB')
        #     np_im = np.array(im)
        #     natural_mask_np_ims.append(np_im)

        #     # Add name to list 
        #     natural_mask_paths.append(picked_path)


        # Pick natural masks
        natural_masks = sorted([path for path in os.listdir(image_dir) if path != '.DS_Store'])
        for mask in natural_masks:

            # im_name = os.path.join(cur_mask_dir, mask)
            # natural_mask_paths.append(im_name)

            im_name = os.path.join(image_dir, mask)

            # Convert image to array and add to list
            im = Image.open(im_name).convert('RGB')

            # np_im = np.array(im)
            # natural_mask_np_ims.append(np_im)
            cropped_path = crop_image(image = (im, mask), root = mask_dir)
            cropped_paths.append(cropped_path)

            cropped_im = Image.open(cropped_path).convert('RGB')
            np_cropped = np.array(cropped_im)
            cropped_mask_np_ims.append(np_cropped) 


        # natural mask
        all_masks['natural'] = cropped_paths

        # scrambled mask v1
        scrambled_p01 = scramble_images_v1(cropped_mask_np_ims[0:5], root = mask_dir, rescale = 'range', p = 0.1) 
        scrambled_p02 = scramble_images_v1(cropped_mask_np_ims[0:5], root = mask_dir, rescale = 'range', p = 0.2) 
        scrambled_p03 = scramble_images_v1(cropped_mask_np_ims[0:5], root = mask_dir, rescale = 'range', p = 0.3) 
        scrambled_p04 = scramble_images_v1(cropped_mask_np_ims[0:5], root = mask_dir, rescale = 'range', p = 0.4) 
        scrambled_p05 = scramble_images_v1(cropped_mask_np_ims[0:5], root = mask_dir, rescale = 'range', p = 0.5) 
        scrambled_p1 = scramble_images_v1(cropped_mask_np_ims[0:5], root = mask_dir, rescale = 'range', p = 1) 

        # scrambled mask v2
        scrambled_p01 = scramble_images_v2(cropped_mask_np_ims[0:5], root = mask_dir, rescale = 'range', p = 0.1) 
        scrambled_p02 = scramble_images_v2(cropped_mask_np_ims[0:5], root = mask_dir, rescale = 'range', p = 0.2) 
        scrambled_p03 = scramble_images_v2(cropped_mask_np_ims[0:5], root = mask_dir, rescale = 'range', p = 0.3) 
        scrambled_p04 = scramble_images_v2(cropped_mask_np_ims[0:5], root = mask_dir, rescale = 'range', p = 0.4) 
        scrambled_p05 = scramble_images_v2(cropped_mask_np_ims[0:5], rescale = 'range', p = 0.5) 
        scrambled_p1 = scramble_images_v2(cropped_mask_np_ims[0:5], root = mask_dir, rescale = 'range', p = 1) 

        # noise mask
        noise_mask_b0 = noise_masks(n = 5, root = mask_dir, image_dimensions= (480, 480), beta = 0)
        noise_mask_b05 = noise_masks(n = 5, root = mask_dir, image_dimensions= (480, 480), beta = 0.5)
        noise_mask_b1 = noise_masks(n = 5, root = mask_dir, image_dimensions= (480, 480), beta = 1)
        noise_mask_b15 = noise_masks(n = 5, root = mask_dir, image_dimensions= (480, 480), beta = 1.5)
        noise_mask_b2 = noise_masks(n = 5, root = mask_dir, image_dimensions= (480, 480), beta = 2)
        noise_mask_b25 = noise_masks(n = 5, root = mask_dir, image_dimensions= (480, 480), beta = 2.5)

        # geometric mask
        geometric_d30 = geometric_masks(sz_y = 480, sz_x = 480, n_masks = 5, root = mask_dir,  d = 30)
        geometric_d60 = geometric_masks(sz_y = 480, sz_x = 480, n_masks = 5, root = mask_dir, d = 60)
        geometric_d120 = geometric_masks(sz_y = 480, sz_x = 480, n_masks = 5, root = mask_dir, d = 120)
        geometric_d240 = geometric_masks(sz_y = 480, sz_x = 480, n_masks = 5, root = mask_dir, d = 240)

        # line mask = 
        lines_200 = line_masks(sz_y = 480, sz_x = 480, n_masks = 5, root = mask_dir, d = 200)
        lines_300 = line_masks(sz_y = 480, sz_x = 480, n_masks = 5, root = mask_dir, d = 300)
        lines_400 = line_masks(sz_y = 480, sz_x = 480, n_masks = 5, root = mask_dir, d = 400)
        lines_500 = line_masks(sz_y = 480, sz_x = 480, n_masks = 5, root = mask_dir, d = 500)

        # block mask
        blocked_tiny = block_scrambled(n_masks = 5, root = mask_dir, target_size = (480, 480), block_size= (30, 30))
        blocked_small = block_scrambled(n_masks = 5, root = mask_dir, target_size = (480, 480), block_size= (60, 60))
        blocked_medium = block_scrambled(n_masks = 5, root = mask_dir, target_size = (480, 480), block_size= (120, 120))
        blocked_large  = block_scrambled(n_masks = 5, root = mask_dir, target_size = (480, 480), block_size= (240, 240))


    if type == 'experiment':

        experiment_dir = os.path.join(wd, 'stimuli', 'experiment')
        mask_dir = os.path.join(experiment_dir, 'masks')

        nat_mask_dir = os.path.join(mask_dir, '1_natural')
        if not os.path.exists(nat_mask_dir):
            os.makedirs(nat_mask_dir)   
          
        # # Pick target concepts
        # target_concepts = []
        # for cat in target_category:
        #     concepts = things_concept['uniqueID'][things_concept['All Bottom-up Categories'].str.contains(cat)]
        #     concepts = random.sample(concepts.values.tolist(), nr_concepts)
        #     target_concepts = target_concepts + concepts

        natural_mask_paths = []
        natural_mask_np = []

        # Pick  natural mask concepts
        mask_concepts = []
        for cat in mask_category:
            concepts = things_concept['uniqueID'][things_concept['All Bottom-up Categories'].str.contains(cat)]
            concepts = random.sample(concepts.values.tolist(), nr_concepts)
            mask_concepts = mask_concepts + concepts

        # Pick target images per concept
        for concept in mask_concepts:

            paths = image_paths[image_paths.iloc[:, 0].str.contains(concept)].values.tolist()

            # Correct for substrings
            corrected_paths = []
            for path in paths:
                if path[0].split('/')[1] == concept:
                    corrected_paths.append(path[0])

            # Pick one image per concept
            picked_path = random.choice(corrected_paths)[7:]
            image_name = picked_path.split('/')[1]
            
            # Get image from image base and convert 
            im_path = os.path.join(wd, 'image_base', picked_path)
            im = Image.open(im_name).convert('RGB')

            # Crop
            cropped_path = crop_image(image = (im, image_name), root = mask_dir)
            cropped_im = Image.open(cropped_path).convert('RGB')

            # Save array
            np_cropped = np.array(cropped_im)
            natural_mask_np.append(np_cropped) 

            # Save selected image at new location
            new_im_name = os.path.join(nat_mask_dir, image_name) # not whole picked path
            natural_mask_paths.append(new_im_name)

        # natural mask
        all_masks['natural'] = natural_mask_paths

        # block mask
        blocked_medium = block_scrambled(n_masks = 20, root = mask_dir, target_size = (480, 480), block_size= (120, 120)) # check file path

        # scamble mask
        all_masks['scrambled'] = scramble_images_v1(natural_mask_np, root = mask_dir, rescale = 'range', p = 0.5) 

        # noise mask
        all_masks['noise'] = noise_masks(n = 20, root = mask_dir, image_dimensions= (480, 480), beta = 2)

        # geometric mask
        all_masks['geometric'] = geometric_masks(sz_y = 480, sz_x = 480, n_masks = 20, root = mask_dir, d = 300)

        # line mask =
        all_masks['lines'] = line_masks(sz_y = 480, sz_x = 480, n_masks = 20, root = mask_dir, d = 300)



def create_selection_csv(self, things_concept = things_concept, image_paths = image_paths, target_category = target_category):
    """creates csv that contains info for all trials:
    - imageID (path to target image)
    - concept
    - category
    - mask (path to mask)"""

    experiment_dir = os.path.join(wd, 'stimuli', 'experiment')
    image_dir = os.path.join(experiment_dir, 'images')
    mask_dir = os.path.join(experiment_dir, 'masks')

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)   

    df = pd.DataFrame(columns= ['ImageID','concept','category', 'mask_type', 'mask_path'])

    # Find according image paths for categories
    for category in target_category:

        concepts = things_concept['uniqueID'][things_concept['All Bottom-up Categories'].str.contains(category)].values.tolist()

        # pick nr_concepts
        concepts = random.sample(concepts, nr_concepts)

        for concept in concepts: 
            
            paths = image_paths[image_paths.iloc[:, 0].str.contains(concept)].values.tolist()

            # Correct for substrings
            corrected_paths = []
            for path in paths:
                if path[0].split('/')[1] == concept:
                    corrected_paths.append(path[0])

            # pick nr_per_concept
            corrected_paths  = random.sample(corrected_paths, nr_per_concept)

            # Save cropped target images to correct path & loop through all target images
            for path in corrected_paths:

                image_name = path[7:]
        
                # Get image from image base and convert 
                im_path = os.path.join(wd, 'image_base', path[7:])
                im = Image.open(im_path).convert('RGB')

                # Crop
                cropped_path = crop_image(image = (im, image_name), root = image_dir)

                # Link masks
                for mask in masks:

                    # find paths of all mask images of that type
                    cur_mask_dir = os.path.join(mask_dir, mask)
                    mask_paths = [path for path in os.listdir(cur_mask_dir) if path != '.DS_Store']

                    for mask_path in mask_paths:

                        info = [cropped_path, concept, category, mask, mask_path]
                        repeated_info = [info for i in range(nr_repeats)]
                        sub_df = pd.DataFrame(repeated_info, columns= ['ImageID','concept','category', 'mask_type', 'mask_path'])

                        # Add to overall df
                        df = pd.concat([df, sub_df])

            # # Link masks
            # for mask in masks:
            #     for i in range(nr_mask_instances):
            #         sub_df = pd.DataFrame(columns= ['ImageID','concept','category', 'mask'])
            #         sub_df['ImageID'] = np.array(corrected_paths)
            #         sub_df['concept'] = np.repeat(concept, len(corrected_paths))
            #         sub_df['category'] = np.repeat(category, len(corrected_paths))
            #         sub_df['mask'] = np.repeat(f"{mask}_{i}", len(corrected_paths)) # change to mask path

            #         for j in range(nr_repeats):
            #             df = pd.concat([df, sub_df])

    # Export 
    df = df.reset_index()
    df = df.drop(columns=['index'])
    df.to_csv(os.path.join(wd, 'help_files', 'selection_THINGS.csv'))  
    

# Main
create_masks(things_concept = things_concept, image_paths = image_paths, mask_category=mask_category, type = 'experiment')
create_selection_csv(things_concept = things_concept, image_paths = image_paths, target_category = target_category)