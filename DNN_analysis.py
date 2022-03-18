"""
DNN analysis file using THINGsvision to extract feature activations for:
- THINGS images
- different types of masks

by Anne Zonneveld, March 2022
"""
from configparser import NoSectionError
import torch
import tensorflow
import os
import re
import thingsvision.vision as vision
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import rsatoolbox
from scipy import stats


from thingsvision.model_class import Model
from typing import Any, Dict, List

# helper functions
def extract_features(
                    model: Any,
                    module_name: str,
                    image_path: str,
                    out_path: str,
                    batch_size: int,
                    flatten_activations: bool,
                    apply_center_crop: bool,
                    clip: bool=False,
) -> np.ndarray:
    """Extract features for a single layer."""
    dl = vision.load_dl(
                        root=image_path,
                        out_path=out_path,
                        batch_size=batch_size,
                        transforms=model.get_transformations(apply_center_crop=apply_center_crop),
                        backend=backend,
    )
    # exctract features
    features, _ = model.extract_features(
                                        data_loader=dl,
                                        module_name=module_name,
                                        flatten_acts=flatten_activations,
                                        clip=clip,
                                        return_probabilities=False,
    )
    return features

def get_module_names(modules: List[Any]) -> List[str]:
    """Yield module names associated with layers."""
    return list(map(lambda m: m.name, modules))

def extract_all_layers(
                        model_name: str,
                        model: Any,
                        image_path: str,
                        out_path: str,
                        batch_size: int,
                        flatten_activations: bool,
                        apply_center_crop: bool,
                        layer: str='conv',
                        clip: bool=False,
) -> Dict[str, np.ndarray]:
    """Extract features for all selected layers and save them to disk."""
    features_per_layer = {}
    module_names = get_module_names(model.model.layers)
    for l, module_name in enumerate(module_names, start=1):
        if re.search(f'{layer}', module_name):
            # extract features for layer "module_name"
            features = extract_features(
                                        model=model,
                                        module_name=module_name,
                                        image_path=image_path,
                                        out_path=out_path,
                                        batch_size=batch_size,
                                        flatten_activations=flatten_activations,
                                        apply_center_crop=apply_center_crop,
                                        clip=clip,
            )
            # replace with e.g., [f'conv_{l:02d}'] or [f'fc_{l:02d}']
            features_per_layer[f'layer_{l:02d}'] = features
            # save features to disk
            vision.save_features(features, f'{out_path}/features_{model_name}_{module_name}', 'npy')
    return features_per_layer


# set input and output paths
wd = '/Users/AnneZonneveld/Documents/STAGE/masking-project/'
#image_dir = os.path.join(wd, 'stimuli/images')
image_dir = os.path.join(wd, 'stimuli/DNN_analysis/masks')

output_dir = os.path.join(wd, 'features')
if not os.path.exists(output_dir):
            os.makedirs(output_dir)    

rdm_dir = os.path.join(output_dir, 'rdms')
if not os.path.exists(rdm_dir):
            os.makedirs(rdm_dir)  

# determine labels for all input images
labels = []
mask_categories = sorted([path for path in os.listdir(image_dir) if path != '.DS_Store'])
for cat in mask_categories:
    dir = os.path.join(image_dir, cat)
    items = sorted([path for path in os.listdir(dir) if path != '.DS_Store'], key=str)
    labels.append(items)
labels = [item for sublist in labels for item in sublist]

# set variables
backend = 'tf' 
pretrained = True 
model_path = None 
batch_size = 32 
apply_center_crop = False 
flatten_activations = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# select model
model_name = 'ResNet50'
model = Model(
            model_name,
            pretrained=pretrained,
            model_path=model_path,
            device=device,
            backend=backend,
)

# select layer --> could do multiple
# module_name = model.show() 
module_names= ['pool1_pool', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
# module_name = 'pool1_pool'

all_features = {}
all_rdms = {}

for module_name in module_names:

    # extract features (single layer)
    features = extract_features(
                                model=model,
                                module_name=module_name,
                                image_path=image_dir,
                                out_path=output_dir,
                                batch_size=batch_size,
                                flatten_activations=flatten_activations,
                                apply_center_crop=apply_center_crop,
                                clip=False,
    )

    all_features.update({f'{module_name}': features})

    # compute representational dissimilarity matrix
    rdm = vision.compute_rdm(features, method='correlation')
    all_rdms.update({f'{module_name}': rdm})

    # plot rdm - matplot lib
    fig  = plt.figure()
    plt.imshow(rdm)
    plt.xlabel("Mask nr", fontsize=15)
    plt.ylabel("Mask nr", fontsize=15)
    plt.yticks(np.arange(rdm.shape[0]), labels)
    plt.rc('ytick', labelsize=8) 
    plt.title(f"Pearson-R RDM {module_name}", fontsize=18)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Dissimilarity', fontsize=15)
    plt.tight_layout()

    rdm_filename = os.path.join(rdm_dir, f'rdm_{module_name}.png')
    fig.savefig(rdm_filename)

# Create dir
develop_dir = os.path.join(output_dir, 'develop_plot')
if not os.path.exists(develop_dir):
            os.makedirs(develop_dir)  


# Define nr of masks
natural_masks = [path for path in os.listdir(os.path.join(image_dir, '1_natural')) if path != '.DS_Store']
total_natural_images = len(natural_masks) 

scrambled_masks_v1 = [path for path in os.listdir(os.path.join(image_dir, '2_scrambled')) if path != '.DS_Store' and 'v1' in path]
n_scrambled_masks_v1 = len(scrambled_masks_v1)
scramble_labels = ['p=0.1','p=0.2', 'p=0.3', 'p=0.4', 'p=0.5', 'p=1']

scrambled_masks_v2 = [path for path in os.listdir(os.path.join(image_dir, '2_scrambled')) if path != '.DS_Store' and 'v2' in path]
n_scrambled_masks_v2 = len(scrambled_masks_v2)

noise_masks = [path for path in os.listdir(os.path.join(image_dir, '3_noise')) if path != '.DS_Store']
n_noise_masks = len(noise_masks)
noise_labels = ['b=0','b=0.5', 'b=1', 'b=1.5', 'b=2', 'b=2.5']

geometric_masks = [path for path in os.listdir(os.path.join(image_dir, '4_geometric')) if path != '.DS_Store']
n_geometric_masks = len(geometric_masks)
geometric_labels = ['d=30', 'd=60', 'd=120', 'd=240']

line_masks = [path for path in os.listdir(os.path.join(image_dir, '5_lines')) if path != '.DS_Store']
n_line_masks = len(line_masks)
line_labels = ['d=200', 'd=300', 'd=400', 'd=500']

block_masks = [path for path in os.listdir(os.path.join(image_dir, '6_blocked')) if path != '.DS_Store']
n_block_masks = len(block_masks)
block_labels = ['d=4', 'd=16', 'd=64', 'd=256']

# total_n_masks = total_natural_images + n_scrambled_masks_v1 + n_scrambled_masks_v2 + n_geometric_masks + n_noise_masks 
total_n_masks = total_natural_images + n_scrambled_masks_v1 + n_scrambled_masks_v2 + n_geometric_masks + n_noise_masks + n_block_masks + n_line_masks

# Keep track of max min values
overall_max = None
overall_min  = None

# Predfine result matrices
mean_natural = np.zeros(shape=(1, 5, total_natural_images - 1))
sem_natural = np.zeros(shape=(1, 5, total_natural_images - 1))
mean_scrambled_v1 = np.zeros(shape=(5, len(scramble_labels), total_natural_images))
sem_scrambled_v1 = np.zeros(shape=(5, len(scramble_labels), total_natural_images))
mean_scrambled_v2 = np.zeros(shape=(5, len(scramble_labels), total_natural_images))
sem_scrambled_v2 = np.zeros(shape=(5, len(scramble_labels), total_natural_images))
mean_noise = np.zeros(shape=(5, len(scramble_labels), total_natural_images))
sem_noise= np.zeros(shape=(5, len(scramble_labels), total_natural_images))
mean_geometric = np.zeros(shape=(5, len(geometric_labels), total_natural_images))
sem_geometric = np.zeros(shape=(5, len(geometric_labels), total_natural_images))
mean_lines = np.zeros(shape=(5, len(line_labels), total_natural_images))
sem_lines = np.zeros(shape=(5, len(line_labels), total_natural_images))
mean_block = np.zeros(shape=(5, len(block_labels), total_natural_images))
sem_block = np.zeros(shape=(5, len(block_labels), total_natural_images))

# Loop through rows (natural images) of RDM
for row in range(total_natural_images - 1):
    
    target_index = row


    # Create array with all relevant correlations for specific row (target image) for all relevant layers (upper right triangle) 
    development_info = np.zeros(shape=(len(module_names), rdm.shape[0] - target_index))
    i = 0
    for key in all_rdms:
        #development_info[i, :] = all_rdms[key][0, :]
        development_info[i, :] = all_rdms[key][row, target_index:rdm.shape[0]]
        i += 1

    # max_diss = np.max(development_info)
    # min_diss = np.min(development_info[:, 1: len(development_info)]) # exclude value 0 since self-correlation

    # # set overall max / min
    # if row == 0:
    #     overall_max = max_diss
    #     overall_min = min_diss
    # elif max_diss > overall_max:
    #     overall_max = max_diss
    # elif min_diss <  overall_min:
    #     overall_min = min_diss

    # Create categorical develop plot

    # if target_index != total_natural_images - 1:
    #     n_natural_masks = total_natural_images - 1 - target_index
    #     natural_masks_x = development_info[:, develop_index: develop_index + n_natural_masks -1 ]        
    #     mean_natural[:,:, target_index] = natural_masks_x.mean(axis = 1)

    # Track relevant point in RDM
    develop_index = 1

    n_natural_masks = total_natural_images - 1 - target_index
    natural_masks_x = development_info[:, develop_index: develop_index + n_natural_masks] 
    # natural_masks_x = development_info[:, develop_index + target_index: total_natural_images] 
    print(f'target index {target_index}')
    print(f'slice : {natural_masks_x.shape}')
    print(f'mean {natural_masks_x.mean(axis = 1)}')
    print(f'sem {stats.sem(natural_masks_x, axis=1)}')
    mean_natural[:,:, target_index] = natural_masks_x.mean(axis = 1)
    sem_natural[:,:, target_index] = stats.sem(natural_masks_x, axis=1)

    if target_index < total_natural_images - 2:
        mean_natural[:,:, target_index] = natural_masks_x.mean(axis = 1)
        sem_natural[:,:, target_index] = stats.sem(natural_masks_x, axis=1)
    else:
        # mean_natural[:,:, target_index] = natural_masks_x
        mean_natural[:,:, target_index] = natural_masks_x.mean(axis = 1)
        sem_natural[:,:, target_index] = np.zeros(shape=(1,5))

    min_diss = np.min(mean_natural)
    max_diss = np.max(mean_natural)

    # set overall max / min
    if row == 0:
        overall_max = max_diss
        overall_min = min_diss
    elif max_diss > overall_max:
        overall_max = max_diss
    elif min_diss <  overall_min:
        overall_min = min_diss

    # Move to next mask cat
    develop_index = develop_index + n_natural_masks
    # develop_index =  develop_index + total_natural_images - 1

    # Scrambled v1 info
    scrambled_masks_v1_x = development_info[:, develop_index:develop_index + n_scrambled_masks_v1] 
    print(f'scrambled v1: {scrambled_masks_v1_x.shape}')

    for i in range(len(scramble_labels)):
        index = np.linspace(i, n_scrambled_masks_v1 - len(scramble_labels) + i - 1, num=int(n_scrambled_masks_v1/len(scramble_labels))).astype(int)
        mean_scrambled_v1[:, i, row] = scrambled_masks_v1_x[:, index].mean(axis=1)
        sem_scrambled_v1[:, i, row] = stats.sem(scrambled_masks_v1_x[:, index], axis=1)

    # min_diss = np.min(mean_natural)
    # max_diss = np.max(mean_natural)

    # # set overall max / min
    # if max_diss > overall_max:
    #     overall_max = max_diss
    # elif min_diss <  overall_min:
    #     overall_min = min_diss

    # Move to next info
    develop_index = develop_index + n_scrambled_masks_v1

    #  Scrambled v2 info
    scrambled_masks_v2_x = development_info[:, develop_index:develop_index + n_scrambled_masks_v2] 
    print(f'scrambled v2: {scrambled_masks_v2_x.shape}')

    for i in range(len(scramble_labels)):
        index = np.linspace(i, n_scrambled_masks_v2 - len(scramble_labels) + i - 1, num=int(n_scrambled_masks_v2/len(scramble_labels))).astype(int)
        mean_scrambled_v2[:, i, row] = scrambled_masks_v2_x[:, index].mean(axis=1)
        sem_scrambled_v2[:, i, row] = stats.sem(scrambled_masks_v2_x[:, index], axis=1)

    # Move to next info
    develop_index = develop_index + n_scrambled_masks_v2

    # Noise info 
    noise_masks_x = development_info[:, develop_index:develop_index + n_noise_masks] 
    print(f'noise: {noise_masks_x.shape}')

    for i in range(len(noise_labels)):
        index = np.linspace(i, n_noise_masks - len(noise_labels) + i - 1, num=int(n_noise_masks/len(noise_labels))).astype(int)
        mean_noise[:, i, row] = noise_masks_x[:, index].mean(axis=1)
        sem_noise[:, i, row] = stats.sem(noise_masks_x[:, index], axis=1)

    # Move to next info
    develop_index = develop_index + n_noise_masks

    # Geometric info
    geometric_masks_x = development_info[:, develop_index:develop_index + n_geometric_masks] 
    print(f'geometric: {geometric_masks_x.shape}')

    for i in range(len(geometric_labels)):
        index = np.linspace(i, n_geometric_masks - len(geometric_labels) + i - 1, num=int(n_geometric_masks/len(geometric_labels))).astype(int)
        mean_geometric[:, i, row] = geometric_masks_x[:, index].mean(axis=1)
        sem_geometric[:, i, row] = stats.sem(geometric_masks_x[:, index], axis=1)

    # Move to next info
    develop_index = develop_index + n_geometric_masks

    # Line info
    line_masks_x = development_info[:, develop_index:develop_index + n_line_masks] 
    print(f'line: {line_masks_x.shape}')

    for i in range(len(line_labels)):
        index = np.linspace(i, n_line_masks - len(line_labels) + i - 1, num=int(n_line_masks/len(line_labels))).astype(int)
        mean_lines[:, i, row] = line_masks_x[:, index].mean(axis=1)
        sem_lines[:, i, row] = stats.sem(line_masks_x[:, index], axis=1)

    # Move to next info
    develop_index = develop_index + n_line_masks

    # Line info
    block_masks_x = development_info[:, develop_index:develop_index + n_block_masks] 
    print(f'line: {line_masks_x.shape}')

    for i in range(len(block_labels)):
        index = np.linspace(i, n_block_masks - len(block_labels) + i - 1, num=int(n_block_masks/len(block_labels))).astype(int)
        mean_block[:, i, row] = block_masks_x[:, index].mean(axis=1)
        sem_block[:, i, row] = stats.sem(block_masks_x[:, index], axis=1)


# Plot grand mean natural masks develop plot 
fig  = plt.figure()
grand_mean_natural = mean_natural.mean(axis=2)[0]
grand_sem_natural = stats.sem(sem_natural, axis=2)[0]

plt.plot(grand_mean_natural, marker='o', color='blue')
plt.ylim([overall_min - 0.05, overall_max + 0.05])
plt.errorbar(np.arange(len(module_names)),grand_mean_natural, yerr=grand_sem_natural, fmt='o', color='darkblue',
             ecolor='lightblue', capsize=10)
plt.title(f"Dissimilarity target-mask: natural")
plt.xlabel("Network depth", fontsize=15)
plt.xticks(np.arange(len(module_names)), module_names, rotation = 45)
plt.ylabel("Dissimilarity", fontsize=15)
plt.tight_layout()
develop_filename = os.path.join(develop_dir, f'categorical.png')
fig.savefig(develop_filename)


# Plot grand mean scramble v1 plot
fig  = plt.figure()
grand_mean_scrambled_v1 = mean_scrambled_v1.mean(axis=2)
grand_sem_scrambled_v1 = stats.sem(sem_scrambled_v1, axis=2)

plt.plot(grand_mean_scrambled_v1, marker='o')

for i in range(sem_scrambled_v1.shape[1]):
    plt.errorbar(np.arange(len(module_names)), grand_mean_scrambled_v1[:, i], yerr=grand_sem_scrambled_v1[:, i], fmt='o',
             ecolor='lightblue', capsize=10)

plt.ylim([overall_min - 0.05, overall_max + 0.05])
plt.title(f"Dissimilarity target-mask: scrambled v1")
plt.xlabel("Network depth", fontsize=15)
plt.xticks(np.arange(len(module_names)), module_names, rotation = 45)
plt.ylabel("Dissimilarity", fontsize=15)
plt.legend(labels= scramble_labels, loc="upper right")
plt.tight_layout()
develop_filename = os.path.join(develop_dir, f'scrambled_v1.png')
fig.savefig(develop_filename)

# Plot grand mean scramble v2 plot
fig  = plt.figure()
grand_mean_scrambled_v2 = mean_scrambled_v2.mean(axis=2)
grand_sem_scrambled_v2 = stats.sem(sem_scrambled_v2, axis=2)

plt.plot(grand_mean_scrambled_v2, marker='o')

for i in range(sem_scrambled_v2.shape[1]):
    plt.errorbar(np.arange(len(module_names)), grand_mean_scrambled_v2[:, i], yerr=grand_sem_scrambled_v2[:, i], fmt='o',
             ecolor='lightblue', capsize=10)

plt.ylim([overall_min - 0.05, overall_max + 0.05])
plt.title(f"Dissimilarity target-mask: scrambled v2")
plt.xlabel("Network depth", fontsize=15)
plt.xticks(np.arange(len(module_names)), module_names, rotation = 45)
plt.ylabel("Dissimilarity", fontsize=15)
plt.legend(labels= scramble_labels, loc="upper right")
plt.tight_layout()
develop_filename = os.path.join(develop_dir, f'scrambled_v2.png')
fig.savefig(develop_filename)

# Plot grand mean noise plot
fig  = plt.figure()
grand_mean_noise= mean_noise.mean(axis=2)
grand_sem_noise = stats.sem(sem_noise, axis=2)

plt.plot(grand_mean_noise, marker='o')

for i in range(sem_noise.shape[1]):
    plt.errorbar(np.arange(len(module_names)), grand_mean_noise[:, i], yerr=grand_sem_noise[:, i], fmt='o',
             ecolor='lightblue', capsize=10)

plt.ylim([overall_min - 0.05, overall_max + 0.05])
plt.title(f"Dissimilarity target-mask: noise")
plt.xlabel("Network depth", fontsize=15)
plt.xticks(np.arange(len(module_names)), module_names, rotation = 45)
plt.ylabel("Dissimilarity", fontsize=15)
plt.legend(labels= noise_labels, loc="upper right")
plt.tight_layout()
develop_filename = os.path.join(develop_dir, f'noise.png')
fig.savefig(develop_filename)

# Plot grand mean geometric plot
fig  = plt.figure()
grand_mean_geometric = mean_geometric.mean(axis=2)
grand_sem_geometric= stats.sem(sem_geometric, axis=2)

plt.plot(grand_mean_geometric, marker='o')

for i in range(sem_geometric.shape[1]):
    plt.errorbar(np.arange(len(module_names)), grand_mean_geometric[:, i], yerr=grand_sem_geometric[:, i], fmt='o',
             ecolor='lightblue', capsize=10)

plt.ylim([overall_min - 0.05, overall_max + 0.05])
plt.title(f"Dissimilarity target-mask: geometric")
plt.xlabel("Network depth", fontsize=15)
plt.xticks(np.arange(len(module_names)), module_names, rotation = 45)
plt.ylabel("Dissimilarity", fontsize=15)
plt.legend(labels= geometric_labels, loc="upper right")
plt.tight_layout()
develop_filename = os.path.join(develop_dir, f'geometric.png')
fig.savefig(develop_filename)

# Plot grand mean line plot
fig  = plt.figure()
grand_mean_lines = mean_lines.mean(axis=2)
grand_sem_lines = stats.sem(sem_lines, axis=2)

plt.plot(grand_mean_lines, marker='o')

for i in range(sem_lines.shape[1]):
    plt.errorbar(np.arange(len(module_names)), grand_mean_lines[:, i], yerr=grand_sem_lines[:, i], fmt='o',
             ecolor='lightblue', capsize=10)

plt.ylim([overall_min - 0.05, overall_max + 0.05])
plt.title(f"Dissimilarity target-mask: lines")
plt.xlabel("Network depth", fontsize=15)
plt.xticks(np.arange(len(module_names)), module_names, rotation = 45)
plt.ylabel("Dissimilarity", fontsize=15)
plt.legend(labels= line_labels, loc="upper right")
plt.tight_layout()
develop_filename = os.path.join(develop_dir, f'lines.png')
fig.savefig(develop_filename)

# Plot grand mean block plot
fig  = plt.figure()
grand_mean_block = mean_block.mean(axis=2)
grand_sem_block = stats.sem(sem_block, axis=2)

plt.plot(grand_mean_block, marker='o')

for i in range(sem_block.shape[1]):
    plt.errorbar(np.arange(len(module_names)), grand_mean_block[:, i], yerr=grand_sem_block[:, i], fmt='o',
             ecolor='lightblue', capsize=10)

plt.ylim([overall_min - 0.05, overall_max + 0.05])
plt.title(f"Dissimilarity target-mask: blocked")
plt.xlabel("Network depth", fontsize=15)
plt.xticks(np.arange(len(module_names)), module_names, rotation = 45)
plt.ylabel("Dissimilarity", fontsize=15)
plt.legend(labels= block_labels, loc="upper right")
plt.tight_layout()
develop_filename = os.path.join(develop_dir, f'blocked.png')
fig.savefig(develop_filename)


# Grand mean overview plot
fig  = plt.figure()
plt.plot(grand_mean_natural, marker='o', label='natural')
plt.plot(grand_mean_scrambled_v1.mean(axis = 1), marker ='o', label='scrambled v1')
plt.plot(grand_mean_scrambled_v2.mean(axis = 1), marker ='o', label='scrambled v2')
plt.plot(grand_mean_geometric.mean(axis = 1), marker ='o', label='geometric')
plt.plot(grand_mean_noise.mean(axis = 1), marker ='o', label='noise')
plt.plot(grand_mean_lines.mean(axis = 1), marker ='o', label='lines')
plt.plot(grand_mean_block.mean(axis = 1), marker ='o', label='blocked')

plt.errorbar(np.arange(len(module_names)), grand_mean_natural, yerr=grand_sem_natural, fmt='o',
             ecolor='lightblue', capsize=10)
plt.errorbar(np.arange(len(module_names)), grand_mean_scrambled_v1.mean(axis = 1), yerr= stats.sem(grand_sem_scrambled_v1, axis=1), fmt='o',
             ecolor='lightblue', capsize=10)
plt.errorbar(np.arange(len(module_names)), grand_mean_scrambled_v2.mean(axis = 1), yerr=stats.sem(grand_sem_scrambled_v2, axis=1), fmt='o',
             ecolor='lightblue', capsize=10)
plt.errorbar(np.arange(len(module_names)), grand_mean_geometric.mean(axis = 1), yerr=stats.sem(grand_sem_geometric, axis=1), fmt='o',
             ecolor='lightblue', capsize=10)
plt.errorbar(np.arange(len(module_names)), grand_mean_noise.mean(axis = 1), yerr=stats.sem(grand_sem_noise, axis=1), fmt='o',
             ecolor='lightblue', capsize=10)
plt.errorbar(np.arange(len(module_names)), grand_mean_lines.mean(axis = 1), yerr=stats.sem(grand_sem_lines, axis=1), fmt='o',
             ecolor='lightblue', capsize=10)
plt.errorbar(np.arange(len(module_names)), grand_mean_block.mean(axis = 1), yerr=stats.sem(grand_sem_block, axis=1), fmt='o',
             ecolor='lightblue', capsize=10)            
             
plt.title(f"Dissimilarity target-mask")
plt.xlabel("Network depth", fontsize=15)
plt.xticks(np.arange(len(module_names)), module_names, rotation = 45)
plt.ylabel("Dissimilarity", fontsize=15)
plt.legend(loc="upper right")
plt.tight_layout()
develop_filename = os.path.join(develop_dir, f'overview.png')
fig.savefig(develop_filename)