import torch
from torch import optim, nn
from torchvision import models 
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
import os
import re
import glob
import shutil
import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats
from IPython import embed as shell
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

wd = '/home/c11645571/masking-project'
trial_file = pd.read_csv(os.path.join(wd, 'help_files', 'selection_THINGS.csv'))  
concept_selection = pd.read_csv(os.path.join(wd, "help_files", "concept_selection.csv"), sep=';', header=0) 
all_targets = pd.unique(trial_file['ImageID']).tolist()
all_masks = pd.unique(trial_file['mask_path']).tolist()
all_masks = [path for path in all_masks if path != 'no_mask']
all_images = pd.unique(all_targets + all_masks).tolist()

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
model = models.resnet50(pretrained=True)

transform  = T.Compose([
            T.ToPILImage(),
            T.CenterCrop(512),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=0., std=1.) # correct parameters?
        ])


def move_images():
    # Move images to analysis map and create correct file structure

    destination_base = os.path.join(wd, 'analysis/images')
    folders = ['1_natural', '2_scrambled', '4_geometric', '5_lines', '6_blocked']

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


def fit_PCA(n_components):
    """Fit PCA on random not-experimental images of THINGS database"""

    # Find images to fit PCA - fit on 1000 random natural images
    image_base_dir = os.path.join(wd, 'image_base_non_exp')
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
    n = 1000
    i_rand = np.random.randint(0, len(all_PCA_images), n)

    imgs = np.zeros((n, 3, 224, 224))
    img_counter = 0 
    for i in i_rand:
        if img_counter % 100 == 0:
            print(f"Preprocessing PCA image {img_counter}")
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

    pca_fits = {}

    for layer in layers:
        print(f"Fitting PCA {layer}")
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

    del imgs
    
    return PCA_fit_features, pca_fits


def feature_extraction(pca_fits, n_components):
    """Extract features using PCA for experimental images"""
    print("Extracting features")

    # Find all experimental images 
    img_dir = os.path.join(wd, 'analysis/images')
    folders = [f for f in os.listdir(img_dir) if not f.startswith('.')]
    img_paths = []
    for folder in folders:
        
        folder_dir = os.path.join(img_dir, folder)

        if folder in ['5_lines', '4_geometric']:
            for path in os.listdir(folder_dir):
                if not path.startswith('.'):
                    img_path = os.path.join(folder_dir, path)
                    img_paths.append(img_path)
            
        elif folder in ['1_natural', '2_scrambled', '6_blocked']:
            concepts = os.listdir(folder_dir)
            for concept in concepts:
                concept_dir = os.path.join(folder_dir, concept)
                for path in os.listdir(concept_dir):
                    if not path.startswith('.'):
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
    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']

    PCA_features = {
        'layer1': np.zeros((len(img_paths), n_components)),
        'layer2': np.zeros((len(img_paths), n_components)),
        'layer3': np.zeros((len(img_paths), n_components)),
        'layer4': np.zeros((len(img_paths), n_components)),
        'layer5': np.zeros((len(img_paths), n_components))
    }

    # Loop through batches
    nr_batches = 5
    batches = np.linspace(0, len(img_paths), nr_batches + 1, endpoint=True, dtype=int)
    
    for b in range(nr_batches):
        print('Processing batch ' + str(b + 1))

        batch_size = batches[b+1] - batches[b]
        imgs = np.zeros((batch_size, 3, 224, 224))

        img_counter = 0 

        # Loop through images batch
        for i in range(batches[b], batches[b+1]):

            if img_counter % 50 == 0:
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
            features = np.reshape(feature_dict[layer].detach().numpy(),(feature_dict[layer].detach().numpy().shape[0], -1)) # flatten
            pca = pca_fits[layer]
            features_pca = pca.transform(features)
            PCA_features[layer][batches[b]:batches[b+1], :] = features_pca

        del features, features_pca, imgs
    
    # Save features
    for layer in layers:
        output_dir = os.path.join(wd, f'analysis/features/{layer}')
        features = PCA_features[layer]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
            with open(os.path.join(output_dir, 'features.npy'), 'wb') as f:
                np.save(f, features)

    return PCA_features

def load_features():
    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']

    features = {}
    output_dir = os.path.join(wd, f'analysis/features/')
    for layer in layers:
        file = os.path.join(output_dir, f'{layer}/features.npy')
        features[layer] = np.load(file)
    
    return features


def decode_animacy(exp_features):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import LeaveOneOut

    # Make labels
    concepts = pd.unique(concept_selection['concept']).tolist()
    animacy_labels = []
    for image in all_images:
        if image.split('/')[-2] in concepts and image.split('/')[-3] == '1_natural':
            # 1 = animate, 0 = inanimate
            animacy_label = concept_selection[concept_selection['concept'] == image.split('/')[-2]]['animate'].values[0]
            animacy_labels.append(animacy_label)
        else:
            # 2 = other
            animacy_labels.append(2)

    animacy_df = {}
    animacy_df['image'] = all_images
    animacy_df['animate'] = animacy_labels
    animacy_df['features'] = exp_features['layer5']

    LDA_model = LinearDiscriminantAnalysis(n_components=2)
    X = animacy_df['features']
    y = np.asarray(animacy_df['animate'])
    LDA_model.fit(X,y)

    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) # change params
    cv = LeaveOneOut()
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

    return

shell()

def preprocess_bdata(): # to do 

    file_path = os.path.join(wd, 'data/')
    files = glob.glob(os.path.join(file_path, '*.pkl'))
    
    data = pd.DataFrame()
    for file in files:
        print(f"processing {file}")

        tmp = joblib.load(file)
        df = pd.DataFrame(tmp['parameterArray'])

        # check for nr of trials
        if len(df) != 1944:
            print(f'{file} incorrect nr of trials')

        # check for duplicates
        duplicates = df.duplicated()
        nr_duplicates = df.duplicated().sum()
        
        if nr_duplicates > 0:

            # Where are duplicates
            print(f'{file} contains duplicates: {nr_duplicates}')
            duplicates = np.array(duplicates)
            index_duplicates = np.where(duplicates == True)
            print(f"location: {index_duplicates}")

            # Any missing trials?
            missing = []
            for i in range(1944):
                try:
                    df[df['index']==i]
                except:
                    print(f"except: {i}")
                    missing.append(i)
            print(f"missing: {missing}")

            # Delete duplicate 
            df = df.drop(labels=index_duplicates[0], axis=0)

        data = pd.concat([data, df], ignore_index=True)

    masks_ordered = ['no_mask', '2_scrambled', '5_lines', '6_blocked', '4_geometric', '1_natural']
    data["mask_type"] = pd.Categorical(data["mask_type"], masks_ordered)


# MAIN
n_components = 500
pca_features, pca_fits = fit_PCA(n_components)
exp_features = feature_extraction(pca_fits, n_components)
# exp_features = load_features()
decode_animacy(exp_features)




