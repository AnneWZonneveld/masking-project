from configparser import NoSectionError
import torch
import torch.nn as nn
# import tensorflow
import os
import re
import shutil
# import thingsvision.vision as vision
import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
# import rsatoolbox
from scipy import stats
from IPython import embed as shell
from sklearn.decomposition import PCA

# from thingsvision.model_class import Model
from typing import Any, Dict, List

wd = '/Users/AnneZonneveld/Documents/STAGE/masking-project/'
trial_file = pd.read_csv(os.path.join(wd, 'help_files', 'selection_THINGS.csv'))  
concept_selection = pd.read_csv(os.path.join(wd, "help_files", "concept_selection.csv"), sep=';', header=0) 
all_targets = pd.unique(trial_file['ImageID']).tolist()
all_masks = pd.unique(trial_file['mask_path']).tolist()
all_masks = [path for path in all_masks if path != 'no_mask']
all_images = pd.unique(all_targets + all_masks).tolist()


# set model variables
backend = 'pt' 
pretrained = True 
model_path = None 
batch_size = 32 
apply_center_crop = True
flatten_activations = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def move_images():
    # Move images to analysis map and create correct file structure

    destination_base = os.path.join(wd, 'analysis/images')
    folders = ['1_natural', '2_scrambled', '4_geometric', '5_lines', '6_blocked']
    # for folder in folders:
    #     if not os.path.exists(os.path.join(destination_base, folder)):
    #         os.makedirs(os.path.join(destination_base, folder))

    for image in all_images:

        # for natural, scrambled and blocked
        mask_type = image.split('/')[-3]
        folder = mask_type

        # for geometric and line masks
        if not mask_type in folders: 
            folder = image.split('/')[-2]
            destination = os.path.join(destination_base, folder)
        else:
            concept = image.split('/')[-2]
            destination = os.path.join(os.path.join(destination_base, folder), concept)

        if not os.path.exists(destination):
            os.makedirs(destination)

        if os.path.isfile(image):

            shutil.copy(image, destination)

                # if mask_type == "6_blocked": # not necessary? 
                #     old_name = os.path.join(destination, image.split('/')[-1])
                #     new_name = os.path.join(destination, image.split('/')[-2] + "_" + image.split('/')[-1]) 
                #     os.rename(old_name, new_name)


# # Helper functions THINGSvision
# def extract_features(
#                     model: Any,
#                     module_name: str,
#                     image_path: str,
#                     out_path: str,
#                     batch_size: int,
#                     flatten_activations: bool,
#                     apply_center_crop: bool,
#                     backend: str,
#                     clip: bool=False,
# ) -> np.ndarray:
#     """Extract features for a single layer."""
#     dl = vision.load_dl(
#                         root=image_path,
#                         out_path=out_path,
#                         batch_size=batch_size,
#                         transforms=model.get_transformations(apply_center_crop=apply_center_crop),
#                         backend=backend,
#     )
#     # exctract features
#     features, _ = model.extract_features(
#                                         data_loader=dl,
#                                         module_name=module_name,
#                                         flatten_acts=flatten_activations,
#                                         clip=clip,
#                                         return_probabilities=False,
#     )
#     return features


# def extract_all_layers(
#                         model_name: str,
#                         model: Any,
#                         image_path: str,
#                         out_path: str,
#                         batch_size: int,
#                         flatten_activations: bool,
#                         apply_center_crop: bool,
#                         layer: Any=nn.Linear,
#                         clip: bool=False,
# ) -> Dict[str, np.ndarray]:
#     """Extract features for all selected layers and save them to disk."""
#     features_per_layer = {}
#     for l, (module_name, module) in enumerate(model.model.named_modules(), start=1):
#         if isinstance(module, layer):
#             # extract features for layer "module_name"
#             features = extract_features(
#                                         model=model,
#                                         module_name=module_name,
#                                         image_path=image_path,
#                                         out_path=out_path,
#                                         batch_size=batch_size,
#                                         flatten_activations=flatten_activations,
#                                         apply_center_crop=apply_center_crop,
#                                         clip=clip,
#             )
#             # replace with e.g., [f'conv_{l:02d}'] or [f'fc_{l:02d}']
#             features_per_layer[f'layer_{l:02d}'] = features
#             # save features to disk
#             vision.save_features(features, f'{out_path}/features_{model_name}_{module_name}', 'npy')
#     return features_per_layer

# def load_features(module_names, model_name, img_dir):
#     # Load model, present images and saves features. 
#     # If features already present, then load features.  

#     wd = '/Users/AnneZonneveld/Documents/STAGE/masking-project/analysis'
#     output_dir = os.path.join(wd, 'features')

#     model = Model(
#                 model_name,
#                 pretrained=pretrained,
#                 model_path=model_path,
#                 device=device,
#                 backend=backend
#     )

#     all_features = {}

#     folder_names = []
#     for module_name in module_names:
#         folder_name = f'{output_dir}/features_{model_name}_{module_name}'
#         folder_names.append(folder_name)

#         if os.path.exists(folder_name):
#             # load existing features
#             print(f"Loading existing features {folder_name}")
#             features = np.load(os.path.join(folder_name, 'features.npy'))
#         else:
#             # extract features (single layer)
#             print(f"Extracting features {folder_name}")
#             features = extract_features(
#                                         model=model,
#                                         module_name=module_name,
#                                         image_path=img_dir,
#                                         out_path=output_dir,
#                                         batch_size=batch_size,
#                                         flatten_activations=flatten_activations,
#                                         apply_center_crop=apply_center_crop,
#                                         clip=False,
#                                         backend = backend
#             )

#             vision.save_features(features, f'{output_dir}/features_{model_name}_{module_name}', 'npy')

#         all_features.update({f'{module_name}': features})

#     return all_features

### MAIN
model_name = 'resnet50'
module_names= ['relu', 'layer1.2.relu', 'layer2.3.relu', 'layer3.5.relu', 'layer4.2.relu']
img_dir = os.path.join(wd, 'analysis/images_test')
# all_features = load_features(module_names = module_names, model_name=model_name, img_dir = img_dir)

# # Loop over modules
# stage1_features = load_features(module_names = ['relu'], model_name=model_name, img_dir = img_dir)


#  ----------------------- Alternative with pytorch
import torch
from torch import optim, nn
from torchvision import models 
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image


class THINGSdataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        image = self.imgs[index]
        X = self.transform(image)
        return X
    
        transform = T.Compose([
            T.ToPILImage(),
            T.CenterCrop(512),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=0., std=1.)
        ])

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # super(FeatureExtractor, self).__init__(model)
        self.stage1 = model.relu
        self.stage2 = [*model.layer1][2].relu
        self.stage3 = [*model.layer2][3].relu
        self.stage4 = [*model.layer3][5].relu
        self.stage5 = [*model.layer4][2].relu

    def forward(self, x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out) 
        out = self.stage4(out)
        return out 


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
model = models.resnet50(pretrained=True)

transform  = T.Compose([
            T.ToPILImage(),
            T.CenterCrop(512),
            T.Resize((224, 224)),
            T.ToTensor(),
            # T.Normalize(mean=0., std=1.)
        ])


# dataset = THINGSdataset(imgs = img_paths)
# dl = DataLoader(dataset, batch_size, shuffle=True, num_workers=3, pin_memory=True)

# fe_model = FeatureExtractor(model)
# fe_model = fe_model.to(device)

# Find images to fit PCA - fit on 1000 random natural images
image_base_dir = '/Volumes/One Touch/STAGE SCHOLTE/image_base_1'
folders = [f for f in os.listdir(image_base_dir) if not f.startswith('.')]
all_PCA_images = []
for folder in folders:
    print(f"Processing folder {folder}")
    concept_dir = os.path.join(image_base_dir, folder)
    images = [f for f in os.listdir(concept_dir) if not f.startswith('.')]
    image_paths = []
    for image in images:
        image_path = os.path.join(concept_dir, image)
        image_paths.append(image_path)
    all_PCA_images.append(image_paths)
all_PCA_images = [item for sublist in all_PCA_images for item in sublist] # flatten list --> 25366 images

return_nodes = {
    # node_name: user-specified key for output dict
    'relu': 'layer1',
    'layer1.2.relu': 'layer2', 
    'layer2.3.relu': 'layer3',
    'layer3.5.relu': 'layer4', 
    'layer4.2.relu': 'layer5'
}

feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

# Preprocess images to fit PCA
n = 10
i_rand = np.random.randint(0, len(all_PCA_images), n)

imgs = np.zeros((n, 3, 224, 224))
img_counter = 0 
for i in i_rand:
    if img_counter % 100 == 0:
        print(f"Preprocessing image {img_counter}")
    img_path = all_PCA_images[i]
    img = np.asarray(Image.open(img_path))
    img = transform(img)
    img = img.reshape(1, 3, 224, 224)
    imgs[img_counter, :, :, :] = img

    img_counter += 1

# Extract features 
imgs = torch.from_numpy(imgs).type(torch.DoubleTensor)
feature_dict_PCA = feature_extractor(imgs) #10 --> not enough RAM

layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']

# Fit PCA --> seperately for every layer
PCA_fit_features = {
    'layer1': np.zeros((n, 64, 112, 112)),
    'layer2': np.zeros((n, 64, 56, 56)),
    'layer3': np.zeros((n, 128, 28, 28)),
    'layer4': np.zeros((n, 256, 14, 14)),
    'layer5': np.zeros((n, 512, 7, 7))
}

n_components = 5 # should be like 500
pca_fits = {}

for layer in layers:
    pca = PCA(n_components=n_components) 
    features = np.reshape(feature_dict_PCA[layer].detach().numpy(),(feature_dict_PCA[layer].detach().numpy().shape[0], -1)) # flatten
    features_pca = pca.fit_transform(features)
    PCA_fit_features[layer] = features_pca
    pca_fits.update({f'{layer}': pca})

# Evaluate how the variance is distributed across the PCA components. 
fig, ax = plt.subplots(1,len(layers), sharey=True)
layer_nr = 0
for layer in layers:
    ax[layer_nr].plot(np.arange(n_components), pca_fits[layer].explained_variance_ratio_, label=layer) # plot all different layers
    ax[layer_nr].set_title(f'{layer} ' + str(round(np.sum(pca_fits[layer].explained_variance_ratio_), 3)))
    layer_nr += 1
fig.supxlabel('Component')
fig.supylabel('Variance explained')
fig.suptitle('Variance explained')
fig.tight_layout()
filename = os.path.join(wd, 'analysis/scree_plot.png')
fig.savefig(filename)


# Find all images - test
img_dir = os.path.join(wd, 'analysis/images_test')
images = os.listdir(img_dir)
img_paths = []
for image in images:
    if image != '.DS_Store':
        img_path = os.path.join(img_dir, image)
        img_paths.append(img_path)

# Find all experimental images 
img_dir = os.path.join(wd, 'analysis/images')
folders = [path for path in os.listdir(img_dir) if path != '.DS_Store']
img_paths = []
for folder in folders:
    
    folder_dir = os.path.join(img_dir, folder)

    if folder in ['5_lines', '4_geometric']:
        for path in os.listdir(folder_dir):
            if path != '.DS_Store':
                img_path = os.path.join(folder_dir, path)
                img_paths.append(img_path)
        
    elif folder in ['1_natural', '2_scrambled', '6_blocked']:
        concepts = os.listdir(folder_dir)
        for concept in concepts:
            concept_dir = os.path.join(folder_dir, concept)
            for path in os.listdir(concept_dir):
                if path != '.DS_Store':
                    img_path = os.path.join(concept_dir, path)
                    img_paths.append(img_path)

# Create text file with all image paths
df = pd.DataFrame (img_paths, columns = ['path'])
df.to_csv(os.path.join(wd, 'analysis', 'image_paths.csv')) 

# Feature extracting
return_nodes = {
    # node_name: user-specified key for output dict
    'relu': 'layer1',
    'layer1.2.relu': 'layer2', 
    'layer2.3.relu': 'layer3',
    'layer3.5.relu': 'layer4', 
    'layer4.2.relu': 'layer5'
}

feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

img_paths = img_paths[0:50] # test

all_features = {
    'layer1': np.zeros((len(img_paths), 64, 112, 112)),
    'layer2': np.zeros((len(img_paths), 64, 56, 56)),
    'layer3': np.zeros((len(img_paths), 128, 28, 28)),
    'layer4': np.zeros((len(img_paths), 256, 14, 14)),
    'layer5': np.zeros((len(img_paths), 512, 7, 7))
}

PCA_features = {
    'layer1': np.zeros((len(img_paths), n_components)),
    'layer2': np.zeros((len(img_paths), n_components)),
    'layer3': np.zeros((len(img_paths), n_components)),
    'layer4': np.zeros((len(img_paths), n_components)),
    'layer5': np.zeros((len(img_paths), n_components))
}

# Loop through batches
nr_batches = 10
batches = np.linspace(0, len(img_paths), nr_batches + 1, endpoint=True, dtype=int)
batch_size = batches[1] - batches[0]

for b in range(nr_batches):
    print('Processing batch ' + str(b + 1))

    imgs = np.zeros((batch_size, 3, 224, 224))

    img_counter = 0 

    # Loop through images batch
    for i in range(batches[b], batches[b+1]):

        if img_counter % 100 == 0:
            print(f"Preprocessing image {img_counter}")

        # Pre process image
        img_path = img_paths[i]
        img = np.asarray(Image.open(img_path))
        img = transform(img)
        img = img.reshape(1, 3, 224, 224)
        imgs[img_counter, :, :, :] = img

        img_counter += 1

    # Extract features 
    imgs = torch.from_numpy(imgs).type(torch.DoubleTensor)
    feature_dict = feature_extractor(imgs)

    # Add to all features
    for layer in layers:
        all_features[layer][batches[b]:batches[b+1], :, :, :] = feature_dict[layer].detach().numpy()    
        features = np.reshape(feature_dict[layer].detach().numpy(),(feature_dict[layer].detach().numpy().shape[0], -1)) # flatten
        pca = pca_fits[layer]
        features_pca = pca.transform(features)
        PCA_features[layer][batches[b]:batches[b+1], :] = features_pca

    del features, features_pca, imgs

# ------------------- LDA to decode animacy - only for last layer 
shell()

# Make labels
all_concepts = pd.unique(concept_selection['concept']).tolist()
animacy_labels = []
for image in all_images:
    if image.split('/')[-2] in concepts and image.split('/')[-3] == '1_natural':
        # 1 = animate, 0 = inanimate
        animacy_label = concept_selection[concept_selection['concept'] == image.split('/')[-2]]['animate'].values[0]
        animacy_labels.append(animacy_label)
    else:
        # 2 = other
        animacy_labels.append(2)

# animacy_df = pd.DataFrame()
# animacy_df['image'] = all_images
# animacy_df['animate'] = animacy_labels
# animacy_df['features'] = PCA_features['layer5']

import random

# Test
animacy_df = {}
animacy_df['image'] = all_images[0:50]
animacy_df['animate'] = random.choices(range(0, 3), k=50)
animacy_df['features'] = PCA_features['layer5']

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

LDA_model = LinearDiscriminantAnalysis(n_components=2)
X = animacy_df['features']
y = np.asarray(animacy_df['animate'])
LDA_model.fit(X,y)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) # change params
scores = cross_val_score(LDA_model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Plot 
data_plot = LDA_model.transform(X)
target_names = ['inanimate', 'animate', 'other']

fig, ax = plt.subplots()
colors = ['red', 'green', 'blue']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(data_plot[y == i, 0], data_plot[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA results')
fig.tight_layout()
filename = os.path.join(wd, 'analysis/LDA_results.png')
fig.savefig(filename)


