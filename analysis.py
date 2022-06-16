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
import pickle as pkl
import json
import joblib
import matplotlib.pyplot as plt
from scipy import stats
from IPython import embed as shell
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from yellowbrick.model_selection import RFECV
from matplotlib.lines import Line2D     
import time 

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
    """Fit PCA on random non-experimental images of THINGS database"""

    return_nodes = {
        # node_name: user-specified key for output dict
        'relu': 'layer1',
        'layer1.2.relu': 'layer2', 
        'layer2.3.relu': 'layer3',
        'layer3.5.relu': 'layer4', 
        'layer4.2.relu': 'layer5'
    }

    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    # Preprocess all experimental images to fit PCA
    n_images = len(all_images)
    imgs = np.zeros((n_images, 3, 224, 224))
    new_paths = []
    for i in range(n_images):
        if i % 100 == 0:
            print(f"Preprocessing PCA image {i}")
        img_path = all_images[i]
        img_path = os.path.join(wd, img_path[53:])
        new_paths.append(img_path)
        img = np.asarray(Image.open(img_path))
        img = transform(img)
        img = img.reshape(1, 3, 224, 224)
        imgs[i, :, :, :] = img
    
    # Create text file with all image paths
    df = pd.DataFrame (new_paths, columns = ['path'])
    df.to_csv(os.path.join(wd, 'analysis', 'image_paths_exp.csv')) 

    # Extract features 
    print(f"Extracting features")
    imgs = torch.from_numpy(imgs).type(torch.DoubleTensor)
    feature_dict_PCA = feature_extractor(imgs) 

    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']

    # Fit PCA --> seperately for every layer
    PCA_features = {
        'layer1': np.zeros((n_images, n_components )),
        'layer2': np.zeros((n_images, n_components )),
        'layer3': np.zeros((n_images, n_components )),
        'layer4': np.zeros((n_images, n_components )),
        'layer5': np.zeros((n_images, n_components ))
    }

    pca_fits = {}

    for layer in layers:
        print(f"Fitting PCA {layer}")
        pca = PCA(n_components=n_components) 
        features = np.reshape(feature_dict_PCA[layer].detach().numpy(),(feature_dict_PCA[layer].detach().numpy().shape[0], -1)) # flatten
        print(f"Tranforming features {layer}")
        features_pca = pca.fit_transform(features)
        PCA_features[layer] = features_pca
        pca_fits.update({f'{layer}': pca})

        del features, features_pca, pca
    
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
    filename = os.path.join(wd, 'analysis/fit_scree_plot.png')
    fig.savefig(filename)

    del imgs
    
    print("Saving fits and features")
    pkl.dump(pca_fits, open(os.path.join(wd, 'analysis/pca_fits.pkl'),"wb"))
    pkl.dump(PCA_features, open(os.path.join(wd, 'analysis/exp_features.pkl'),"wb"))

    # Save features 
    for layer in layers:

        output_dir = os.path.join(wd, f'analysis/exp_features1/{layer}')
        reduced_features = PCA_features[layer]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
            with open(os.path.join(output_dir, 'features.npy'), 'wb') as f:
                np.save(f, reduced_features)

    return PCA_features, pca_fits

def random_fit_PCA(n_components):
    return_nodes = {
        # node_name: user-specified key for output dict
        'relu': 'layer1',
        'layer1.2.relu': 'layer2', 
        'layer2.3.relu': 'layer3',
        'layer3.5.relu': 'layer4', 
        'layer4.2.relu': 'layer5'
    }

    random_model = models.resnet50(pretrained=False)
    feature_extractor = create_feature_extractor(random_model, return_nodes=return_nodes)

    # Preprocess all experimental images to fit PCA
    n_images = len(all_images)
    imgs = np.zeros((n_images, 3, 224, 224))
    new_paths = []
    for i in range(n_images):
        if i % 100 == 0:
            print(f"Preprocessing PCA image {i}")
        img_path = all_images[i]
        img_path = os.path.join(wd, img_path[53:])
        new_paths.append(img_path)
        img = np.asarray(Image.open(img_path))
        img = transform(img)
        img = img.reshape(1, 3, 224, 224)
        imgs[i, :, :, :] = img

    # Extract features 
    print(f"Extracting features")
    imgs = torch.from_numpy(imgs).type(torch.DoubleTensor)
    feature_dict_PCA = feature_extractor(imgs) 

    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']

    # Fit PCA --> seperately for every layer
    PCA_features = {
        'layer1': np.zeros((n_images, n_components )),
        'layer2': np.zeros((n_images, n_components )),
        'layer3': np.zeros((n_images, n_components )),
        'layer4': np.zeros((n_images, n_components )),
        'layer5': np.zeros((n_images, n_components ))
    }

    pca_fits = {}

    for layer in layers:
        print(f"Fitting PCA {layer}")
        pca = PCA(n_components=n_components) 
        features = np.reshape(feature_dict_PCA[layer].detach().numpy(),(feature_dict_PCA[layer].detach().numpy().shape[0], -1)) # flatten
        print(f"Tranforming features {layer}")
        features_pca = pca.fit_transform(features)
        PCA_features[layer] = features_pca
        pca_fits.update({f'{layer}': pca})

        del features, features_pca, pca
    
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
    filename = os.path.join(wd, 'analysis/random_fit_scree_plot.png')
    fig.savefig(filename)

    del imgs
    
    print("Saving fits and features")
    pkl.dump(pca_fits, open(os.path.join(wd, 'analysis/random_pca_fits.pkl'),"wb"))
    pkl.dump(PCA_features, open(os.path.join(wd, 'analysis/random_features.pkl'),"wb"))

    # Save features 
    for layer in layers:

        output_dir = os.path.join(wd, f'analysis/random_features1/{layer}')
        reduced_features = PCA_features[layer]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
            with open(os.path.join(output_dir, 'features.npy'), 'wb') as f:
                np.save(f, reduced_features)

    return PCA_features, pca_fits


def load_pca_fits():
    pca_reload = pkl.load(open(os.path.join(wd, 'analysis/pca_fits.pkl'),'rb'))
    return pca_reload

def feature_extraction_animacy(pca_fits, n_components):
    """Extract features using PCA for only experimental target images"""
    print("Extracting features")

    # Create text file with all image paths
    img_base = os.path.join(wd, 'stimuli/experiment/masks/1_natural')
    img_paths = []
    for path in all_targets:
        img_path = os.path.join(os.path.join(img_base, path.split('/')[-2]), path.split('/')[-1])
        img_paths.append(img_path)

    df = pd.DataFrame (img_paths, columns = ['path'])
    df.to_csv(os.path.join(wd, 'analysis', 'image_paths_animacy.csv')) 

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
            all_features[layer][batches[b]:batches[b+1], :, :, :] =  feature_dict[layer].detach().numpy()

            features = np.reshape(feature_dict[layer].detach().numpy(),(feature_dict[layer].detach().numpy().shape[0], -1)) # flatten
            pca = pca_fits[layer]
            features_pca = pca.transform(features)
            PCA_features[layer][batches[b]:batches[b+1], :] = features_pca

        del features, features_pca, imgs

    # Save features 
    for layer in layers:

        output_dir = os.path.join(wd, f'analysis/animacy_features/{layer}')
        reduced_features = PCA_features[layer]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
            with open(os.path.join(output_dir, 'features.npy'), 'wb') as f:
                np.save(f, reduced_features)
    
    return PCA_features



def load_features(atype = 'exp'): # add type argument
    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']

    features = {}
    if atype == 'exp':
        output_dir = os.path.join(wd, f'analysis/exp_features/')
    elif atype == 'animacy':
        output_dir = os.path.join(wd, f'analysis/animacy_features/')
    elif atype =='random':
        output_dir = os.path.join(wd, f'analysis/random_features/')

    for layer in layers:
        file = os.path.join(output_dir, f'{layer}/features.npy')
        features[layer] = np.load(file)
    
    return features


def decode_animacy(features):
    """Decode animacy with logistic regression for only target images."""

    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn import metrics
    from sklearn import preprocessing
    from sklearn.model_selection import LeaveOneOut
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import classification_report, confusion_matrix

    # Make labels
    file_dir = os.path.join(wd, 'analysis', 'image_paths_animacy.csv')
    image_paths = pd.read_csv(file_dir)['path'].tolist()
    concepts = pd.unique(concept_selection['concept']).tolist()
    
    animacy_labels = []
    for image in image_paths:
        if image.split('/')[-2] in concepts:
            # 1 = animate, 0 = inanimate
            animacy_label = concept_selection[concept_selection['concept'] == image.split('/')[-2]]['animate'].values[0]
            animacy_labels.append(animacy_label)

    animacy_df = {}
    animacy_df['image'] = image_paths
    animacy_df['animate'] = animacy_labels
    animacy_df['features'] = features['layer5']

    X = animacy_df['features']
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    y = np.asarray(animacy_df['animate'])
    logit_model = LogisticRegressionCV().fit(X_scaled,y)

    cv = RepeatedKFold(n_splits=10, n_repeats=3 )# change params
    scores = cross_val_score(logit_model, X_scaled, y, scoring='accuracy', cv=cv, n_jobs=-1)

    # summarize result
    print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    return (np.mean(scores), np.std(scores))

def inspect_predictions():
    imagenet_file = os.path.join(wd, 'help_files/LOC_synset_mapping.txt')
    with open(imagenet_file) as f:
        imagenet_labels = [line.strip() for line in f.readlines()]
    
    img_base = os.path.join(wd, 'stimuli/experiment/masks/1_natural')
    img_paths = []
    for path in all_targets:
        img_path = os.path.join(os.path.join(img_base, path.split('/')[-2]), path.split('/')[-1])
        img_paths.append(img_path)
    
    things_labels = [path.split('/')[-2] for path in img_paths]

    # Loop through batches
    nr_batches = 5
    batches = np.linspace(0, len(img_paths), nr_batches + 1, endpoint=True, dtype=int)
    target_img_t = torch.DoubleTensor()
    
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

        imgs = torch.from_numpy(imgs).type(torch.DoubleTensor)
        target_img_t = torch.cat((target_img_t, imgs), dim=0)
    
    outputs = model(target_img_t)
    _, predicted = torch.max(outputs, 1)

    for i in range(predicted.shape[0]):
        print(f'Actual: {things_labels[i]}, predicted: {imagenet_labels[predicted[i]]}')


def preprocess_bdata(): # to do 
    """Files with duplicates:
    - 10_3_2022-05-19_09.01.36 contains duplicates: 1 (index 2) --> checked
    - 14_1_2022-05-17_09.16.37 contains duplicates: 7 (index 11) --> checked 
    - 15_3_2022-05-19_09.01.13 contains duplicates: 2 (index 16)
    - 3_1_2022-04-22_12.44.28 contains duplicates: 1 (index )
    """
    import joblib

    file_path = os.path.join(wd, 'data/')
    files = sorted(glob.glob(os.path.join(file_path, '*.pkl')))
    
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

    return data

from sklearn.pipeline import make_pipeline, Pipeline
class MyPipeline(Pipeline):
    @property
    def coef_(self):
        return self._final_estimator.coef_
    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_


def logit_model(data, exp_features, random_features):
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    from sklearn.model_selection import RepeatedKFold, StratifiedKFold, GridSearchCV, cross_validate, train_test_split, cross_val_predict, GroupShuffleSplit
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.pipeline import make_pipeline, Pipeline
    import statsmodels.api as sm
    from sklearn import preprocessing
    from yellowbrick.model_selection import RFECV

    file_dir = os.path.join(wd, 'analysis', 'image_paths_exp.csv') # should be cc1
    image_paths = pd.read_csv(file_dir)['path'].tolist()
    concepts = pd.unique(concept_selection['concept']).tolist()

    n_trials = len(trial_file)
    trial_df = pd.DataFrame()
   
    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']
    n_layers = len(layers)
    no_mask = [i for i in range(len(trial_file)) if trial_file.iloc[i]['mask_type']=='no_mask']
    n_mask_trials = n_trials - len(no_mask)

    # all_target_activations = np.zeros((n_mask_trials, n_components, n_layers)) 
    # all_mask_activations = np.zeros((n_mask_trials, n_components, n_layers))

    print("Creating trial df")
    no_mask_id = 0
    mask_trials = []
    for i in range(len(trial_file)):

        tmp = pd.DataFrame()
        # tmp_single = pd.DataFrame()

        trial = trial_file.iloc[i]
        trial_id  = i

        # Check if trial is mask trial
        mask_path = trial['mask_path']

        if mask_path != 'no_mask':
            mask_trials.append(i)
            
            # get target path
            target_path = trial['ImageID']
            target_path = os.path.join(wd, target_path[53:])

            # get according activations
            target_index = image_paths.index(target_path)
            target_activations  = np.zeros((n_components, n_layers))
            rtarget_activations  = np.zeros((n_components, n_layers))
            for i in range(n_layers):
                layer = layers[i]
                target_activation = exp_features[layer][target_index, :]
                target_activations[:, i] = target_activation

                rtarget_activation = random_features[layer][target_index, :]
                rtarget_activations[:, i] = rtarget_activation
                        
            # get mask path
            mask_path = os.path.join(wd, mask_path[53:])

            # get according activation
            mask_index = image_paths.index(mask_path)
            mask_activations  = np.zeros((n_components, n_layers))
            rmask_activations  = np.zeros((n_components, n_layers))
            for i in range(n_layers):
                layer = layers[i]
                mask_activation = exp_features[layer][mask_index, :]
                mask_activations[:, i] = mask_activation

                rmask_activation = random_features[layer][mask_index, :]
                rmask_activations[:, i] = rmask_activation

            # get response for all participants (average?)
            responses = data[data['index'] == trial_id]['answer'].tolist() #anwer or correct?
            subject_nrs = data[data['index'] == trial_id]['subject_nr'].tolist()
            valid = data[data['index'] == trial_id]['valid_cue'].tolist()
            mask_type = mask_path.split('/')[-3]
            if mask_type == 'masks':
                mask_type = mask_path.split('/')[-2]
            
            tmp['index'] = [trial_id for i in range(len(responses))]
            tmp['response'] = responses
            tmp['valid'] = valid
            tmp['subject_nr'] = subject_nrs
            tmp['target_path'] = [target_path for i in range(len(responses))]
            tmp['mask_path'] = [mask_path for i in range(len(responses))]
            tmp['mask_type'] = [mask_type for i in range(len(responses))]
            tmp['mask_activation'] = [mask_activations for i in range(len(responses))]
            tmp['target_activation'] = [target_activations for i in range(len(responses))]
            tmp['rmask_activation'] = [rmask_activations for i in range(len(responses))]
            tmp['rtarget_activation'] = [rtarget_activations for i in range(len(responses))]

            trial_df = pd.concat([trial_df, tmp], ignore_index=True)

            no_mask_id =+ 1

    # Only get valid trials 
    select_trial_df = trial_df[trial_df['valid']==1]

    # Inspect how may valid trials per unique trial
    # select_trial_df.groupby(['index']).count()['response']
    # select_trial_df.groupby(['index']).count()['response'].min() # 8 
    # select_trial_df.groupby(['index']).count()['response'].max() # 30
    
    # Activations for all trials, all ppn
    X1 = np.zeros((len(select_trial_df), n_components, n_layers))
    X2 = np.zeros((len(select_trial_df), n_components, n_layers))

    rX1 =  np.zeros((len(select_trial_df), n_components, n_layers))
    rX2 = np.zeros((len(select_trial_df), n_components, n_layers))

    for i in range(len(select_trial_df)):
            X1[i, :, :] = select_trial_df['target_activation'].iloc[i]
            X2[i, :, :] = select_trial_df['mask_activation'].iloc[i]
            rX1[i, :, :] = select_trial_df['rtarget_activation'].iloc[i]
            rX2[i, :, :] = select_trial_df['rmask_activation'].iloc[i]
    X = np.concatenate((X1, X2), axis=1)
    rX = np.concatenate((rX1, rX2), axis=1)
    y = np.asarray(select_trial_df['response'])

    # Split in development (train) and test / balanced for mask type
    indices = np.arange(len(select_trial_df))
    X_train, X_test, y_train, y_test, out_train_inds, out_test_inds = train_test_split(X, y, indices, test_size=0.1, stratify=select_trial_df['mask_type'].values, random_state=0)

    X_train = X[out_train_inds]
    X_test = X[out_test_inds]
    rX_train = rX[out_train_inds]
    rX_test = rX[out_test_inds]
    y_train = y[out_train_inds]
    y_test = y[out_test_inds]

    # Final evaluation also look at r2 squared + mean square error


    # Accuracy human performance
    select_trial_df.iloc[out_train_inds]['response'].mean() #0.223


    shell()
    train_df_path = os.path.join(wd, 'analysis/lr_train_df.csv')
    train_df = pd.read_csv(train_df_path)


    #  ------------------- Logistic regression -------------------------------------------------------------
    # Inspect data
    # select_trial_df['response'].value_counts(normalize=True)
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    select_trial_df['response'].value_counts(normalize=True).plot.pie()
    fig.add_subplot(1,2,2)
    sns.countplot(x=select_trial_df['response'])
    plt.tight_layout()
    sns.despine(offset=0, trim=True)
    file_name = os.path.join(wd, 'analysis/reponse-balance.png')
    plt.savefig(
        file_name)

    # Simple model (without feature eliminatinon / grid search) + CV
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y=y_train)
    lr_basemodel = LogisticRegression(max_iter=5000, class_weight = {0:class_weights[0], 1:class_weights[1]})
    clf = make_pipeline(preprocessing.StandardScaler(), lr_basemodel)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scoring = ['accuracy', 'f1', 'recall', 'precision']

    # Evaluate lr model for different layers and for random network
    all_scores = []
    all_cms = []
    all_pred = []
    all_pred_ind = []
    all_random_scores = []
    all_random_cms = []
    for i in range(n_layers):
        print(f"{layers[i]}")

        # Imagnet Network Activations - LR
        scores = cross_validate(clf, X_train[:, :, i], y_train, scoring=scoring, cv=cv, return_train_score=True)
        # scores = cross_validate(clf, X_train[:, :, i], y_train, scoring=scoring, cv=cv)
        all_scores.append(scores)
        mean_accuracy = np.mean(scores['test_accuracy'])
        mean_f1 = np.mean(scores['test_f1'])
        mean_recall = np.mean(scores['test_recall'])
        mean_precision = np.mean(scores['test_precision'])
        print(f"Mean accuracy: {mean_accuracy}, std {np.std(scores['test_accuracy'])}") #0.555
        print(f"Mean f1: {mean_f1}, std {np.std(scores['test_f1'])}") #0.557
        print(f"Mean recall: {mean_recall}, std {np.std(scores['test_recall'])} ") #0.476
        print(f"Mean precision: {mean_precision}, std {np.std(scores['test_precision'])}") #0.672

        # Cross validate prediction
        print(f"Cross val prediction")
        boot_res = []
        boot_inds = []
        for boot in range(10):
            print(f"Boot {boot}")
            boot_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)

            boot_test_ind = []
            for train_index, test_index in boot_cv.split(X_train, y_train):
                # boot_test_inds.append(test_index)
                boot_test_ind.append(test_index.tolist())

            boot_test_ind = [x for boot in boot_test_ind for x in boot]
            boot_inds.append(boot_test_ind)

            y_train_pred = cross_val_predict(clf, X_train[:, :, i], y_train, cv=boot_cv)
            boot_res.append(y_train_pred)
        all_pred.append(boot_res)
        all_pred_ind.append(boot_inds)

        # Random network activations - LR
        random_scores = cross_validate(clf, rX_train[:, :, i], y_train, scoring=scoring, cv=cv, return_train_score=True)
        all_random_scores.append(random_scores)
        mean_accuracy = np.mean(random_scores['test_accuracy'])
        mean_f1 = np.mean(random_scores['test_f1'])
        mean_recall = np.mean(random_scores['test_recall'])
        mean_precision = np.mean(random_scores['test_precision'])
        print(f"Mean accuracy random: {mean_accuracy}, std {np.std(random_scores['test_accuracy'])}") #0.555
        print(f"Mean f1 random: {mean_f1}, std {np.std(random_scores['test_f1'])}") #0.557
        print(f"Mean recall random: {mean_recall}, std {np.std(random_scores['test_recall'])} ") #0.476
        print(f"Mean precision random: {mean_precision}, std {np.std(random_scores['test_precision'])}") #0.672

        # # Cross validate prediction
        # print(f"Cross val prediction")
        # ry_train_pred = cross_val_predict(clf, rX_train[:, :, i], y_train, cv=cv)
        # all_random_cms.append(cm)

    print("All layer model")
    # All layer model 
    all_X1 = np.zeros((len(select_trial_df), n_components * n_layers))
    all_X2 = np.zeros((len(select_trial_df), n_components * n_layers))

    all_rX1 = np.zeros((len(select_trial_df), n_components * n_layers))
    all_rX2 = np.zeros((len(select_trial_df), n_components * n_layers))

    for i in range(len(select_trial_df)):
        for layer in range(n_layers):
            all_X1[i, layer * n_components : (layer * n_components) + n_components] = select_trial_df['target_activation'].iloc[i][:, layer]
            all_X2[i, layer * n_components : (layer * n_components) + n_components] = select_trial_df['mask_activation'].iloc[i][:, layer]
            all_rX1[i, layer * n_components : (layer * n_components) + n_components] = select_trial_df['rtarget_activation'].iloc[i][:, layer]
            all_rX2[i, layer * n_components : (layer * n_components) + n_components] = select_trial_df['rmask_activation'].iloc[i][:, layer]

    all_X = np.concatenate((all_X1, all_X2), axis=1)
    all_rX = np.concatenate((all_rX1, all_rX2), axis=1)

    # Split in development (train) and test / balanced for mask type
    all_X_train = all_X[out_train_inds]
    all_X_test = all_X[out_test_inds]
    all_rX_train = all_rX[out_train_inds]
    all_rX_test = all_rX[out_test_inds]

    # Evaluate lr model 
    # Imagnet Network Activations - LR
    scores = cross_validate(clf, all_X_train, y_train, scoring=scoring, cv=cv, return_train_score=True)
    all_scores.append(scores)
    mean_accuracy = np.mean(scores['test_accuracy'])
    mean_f1 = np.mean(scores['test_f1'])
    mean_recall = np.mean(scores['test_recall'])
    mean_precision = np.mean(scores['test_precision'])
    print(f"Mean accuracy: {mean_accuracy}, std {np.std(scores['test_accuracy'])}") #0.555
    print(f"Mean f1: {mean_f1}, std {np.std(scores['test_f1'])}") #0.557
    print(f"Mean recall: {mean_recall}, std {np.std(scores['test_recall'])} ") #0.476
    print(f"Mean precision: {mean_precision}, std {np.std(scores['test_precision'])}") #0.672

    # Cross validate prediction
    print(f"Cross val prediction")
    boot_res = []
    boot_inds = []
    for i in range(10):
        print(f"Boot {i}")
        boot_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)

        boot_test_ind = []
        for train_index, test_index in boot_cv.split(X_train, y_train):
            boot_test_ind.append(test_index.tolist())

        boot_test_ind = [x for boot in boot_test_ind for x in boot]
        boot_inds.append(boot_test_ind)

        y_train_pred = cross_val_predict(clf, all_X_train, y_train, cv=boot_cv)
        boot_res.append(y_train_pred)
    all_pred.append(boot_res)
    all_pred_ind.append(boot_inds)

    # Random network activations - LR
    random_scores = cross_validate(clf, all_rX_train, y_train, scoring=scoring, cv=cv, return_train_score=True)
    all_random_scores.append(random_scores)
    mean_accuracy = np.mean(random_scores['test_accuracy'])
    mean_f1 = np.mean(random_scores['test_f1'])
    mean_recall = np.mean(random_scores['test_recall'])
    mean_precision = np.mean(random_scores['test_precision'])
    print(f"Mean accuracy random: {mean_accuracy}, std {np.std(random_scores['test_accuracy'])}") #0.555
    print(f"Mean f1 random: {mean_f1}, std {np.std(random_scores['test_f1'])}") #0.557
    print(f"Mean recall random: {mean_recall}, std {np.std(random_scores['test_recall'])} ") #0.476
    print(f"Mean precision random: {mean_precision}, std {np.std(random_scores['test_precision'])}") #0.672

    # Accuracy over layers plot
    test_accuracies = []
    test_f1s = []
    test_precisions = []
    test_recalls = []
    train_accuracies = []
    train_f1s = []
    train_precisions = []
    train_recalls = []
    test_r_accuracies = []
    test_r_f1s = []
    test_r_precisions = []
    test_r_recalls = []
    train_r_accuracies = []
    train_r_f1s = []
    train_r_precisions = []
    train_r_recalls = []
    for i in range(len(all_scores)):
        score = all_scores[i]
        r_score = all_random_scores[i]

        test_accuracies.append(np.mean(score['test_accuracy']))
        test_f1s.append(np.mean(score['test_f1']))
        test_precisions.append(np.mean(score['test_precision']))
        test_recalls.append(np.mean(score['test_recall']))

        test_r_accuracies.append(np.mean(r_score['test_accuracy']))
        test_r_f1s.append(np.mean(r_score['test_f1']))
        test_r_precisions.append(np.mean(r_score['test_precision']))
        test_r_recalls.append(np.mean(r_score['test_recall']))

        train_accuracies.append(np.mean(score['train_accuracy']))
        train_f1s.append(np.mean(score['train_f1']))
        train_precisions.append(np.mean(score['train_precision']))
        train_recalls.append(np.mean(score['train_recall']))

        train_r_accuracies.append(np.mean(r_score['train_accuracy']))
        train_r_f1s.append(np.mean(r_score['train_f1']))
        train_r_precisions.append(np.mean(r_score['train_precision']))
        train_r_recalls.append(np.mean(r_score['train_recall']))

    xlabels = layers + ['all']
    score_df = pd.DataFrame()
    score_df['test_accuracy'] = test_accuracies
    score_df['test_f1'] = test_f1s
    score_df['test_precision'] = test_precisions
    score_df['test_recall'] = test_recalls
    score_df['layer'] = xlabels
    score_df['test_r_accuracy'] = test_r_accuracies
    score_df['test_r_f1'] = test_r_f1s
    score_df['test_r_precision'] = test_r_precisions
    score_df['test_r_recall'] = test_r_recalls
    score_df['train_accuracy'] = train_accuracies
    score_df['train_f1'] = train_f1s
    score_df['train_precision'] = train_precisions
    score_df['train_recall'] = train_recalls
    score_df['train_r_accuracy'] = train_r_accuracies
    score_df['train_r_f1'] = train_r_f1s
    score_df['train_r_precision'] = train_r_precisions
    score_df['train_r_recall'] = train_r_recalls

    #  Plot performance over layers
    fig, ax =  plt.subplots(2,2, sharex=True, dpi=100, figsize=(14,7))
    fig.suptitle('Model performance')
    fig.supxlabel('Layer')
    fig.supylabel('Score')
    sns.lineplot(data=score_df, x='layer', y='train_accuracy', color='red', linestyle='--', ax=ax[0,0])
    sns.lineplot(data=score_df, x='layer', y='test_accuracy', color='red', ax=ax[0,0])
    sns.lineplot(data=score_df, x='layer', y='train_r_accuracy', color='firebrick', linestyle='--', ax=ax[0,0])
    sns.lineplot(data=score_df, x='layer', y='test_r_accuracy', color='firebrick', ax=ax[0,0])
    ax[0,0].set_title("Accuracy")
    ax[0,0].yaxis.label.set_visible(False)
    ax[0,0].xaxis.label.set_visible(False)
    ax[0,0].legend(handles=[
        Line2D([], [], marker='_', color="red", label="Pretrained"), 
        Line2D([], [], marker='_', color="firebrick", label="Random")], loc='upper right')
    sns.lineplot(data=score_df, x='layer', y='train_f1', color='blue', linestyle='--', ax=ax[0,1])
    sns.lineplot(data=score_df, x='layer', y='test_f1', color='blue', ax=ax[0,1])
    sns.lineplot(data=score_df, x='layer', y='train_r_f1', color='darkblue', linestyle='--', ax=ax[0,1])
    sns.lineplot(data=score_df, x='layer', y='test_r_f1', color='darkblue', ax=ax[0,1])
    ax[0,1].set_title("F1")
    ax[0,1].yaxis.label.set_visible(False)
    ax[0,1].xaxis.label.set_visible(False)
    ax[0,1].legend(handles=[
        Line2D([], [], marker='_', color="blue", label="Pretrained"),
        Line2D([], [], marker='_', color="darkblue", label="Random")], loc='upper right')
    sns.lineplot(data=score_df, x='layer', y='train_precision', color='palegreen', linestyle='--', ax=ax[1,0])
    sns.lineplot(data=score_df, x='layer', y='test_precision', color='palegreen', ax=ax[1,0])
    sns.lineplot(data=score_df, x='layer', y='train_r_precision', color='darkgreen', linestyle='--', ax=ax[1,0])
    sns.lineplot(data=score_df, x='layer', y='test_r_precision', color='darkgreen', ax=ax[1,0])
    ax[1,0].set_title("Precision")
    ax[1,0].yaxis.label.set_visible(False)
    ax[1,0].xaxis.label.set_visible(False)
    ax[1,0].set_xticklabels([1,2,3,4,5])
    ax[1,0].legend(handles=[
        Line2D([], [], marker='_', color="palegreen", label="Pretrained"),
        Line2D([], [], marker='_', color="darkgreen", label="Random")], loc='upper right')
    sns.lineplot(data=score_df, x='layer', y='train_recall', color='peachpuff', linestyle='--', ax=ax[1,1])
    sns.lineplot(data=score_df, x='layer', y='test_recall', color='peachpuff', ax=ax[1,1])
    sns.lineplot(data=score_df, x='layer', y='train_r_recall', color='darkorange', linestyle='--', ax=ax[1,1])
    sns.lineplot(data=score_df, x='layer', y='test_r_recall', color='darkorange', ax=ax[1,1])
    ax[1,1].set_title("Recall")
    ax[1,1].yaxis.label.set_visible(False)
    ax[1,1].xaxis.label.set_visible(False)
    ax[1,1].set_xticklabels(["1","2","3","4", "5", "all"])
    ax[1,1].legend(handles=[        
        Line2D([], [], marker='_', color="peachpuff", label="Pretrained"),
        Line2D([], [], marker='_', color="darkorange", label="Random")], loc='upper right')

    plt.figlegend(handles = [
        Line2D([], [], marker='_', color="black", label="Test score"),
        Line2D([], [], marker='_', color="black", label="Training score", linestyle="--")
        ])
    fig.tight_layout()
    file_name = os.path.join(wd, 'analysis/TrainTest_Score_across_layers.png')
    fig.savefig(file_name)  


    # Only do this, when train df is not present yet
    # Inspect predictions --> bootstrap
    select_trial_df = select_trial_df.reset_index()
    # determine train df
    train_df = select_trial_df.iloc[out_train_inds]
    train_df = train_df.drop('level_0', 1)
    train_df = train_df.reset_index()

    shell()
    # Finding correct predictions
    print("Finding correct predictions")
    for i in range(len(all_pred)):
        layer = xlabels[i]
        print(f"{layer}")
        layer_preds = all_pred[i]
        boot_inds = all_pred_ind[i]

        for boot in range(10):
            layer_pred = layer_preds[boot]
            boot_ind = boot_inds[boot]
            order_pred = []
            for j in range(len(train_df)):
                idx = np.where(np.array(boot_ind) == j)
                order_pred.append(layer_pred[idx][0])

            train_df[layer + f'_pred{boot}'] = order_pred

    # Save train_df
    train_df.to_csv(os.path.join(wd, 'analysis', 'lr_train_df.csv')) 
    
    # Calculate accuracy
    print("Calculating prediction accuracies")
    accuracies = {}
    std_accuracies = {}
    n_masks = len(pd.unique(select_trial_df['mask_type']))
    for layer in xlabels:
        print(f"{layer}")
        boot_accuracies = np.zeros((10, n_masks))
        for boot in range(10):
            print(f"Boot {boot}")
            correct = []
            for i in range(len(train_df)):
                response = train_df.iloc[i]['response']
                prediction = train_df.iloc[i][layer + f'_pred{boot}']
                if response == prediction:
                    correct.append(1)
                else:
                    correct.append(0)

            train_df[layer + f'_correct{boot}'] = correct
            accuracy_boot = train_df.groupby('mask_type')[layer + f'_correct{boot}'].sum() / train_df.groupby('mask_type')[layer + f'_correct{boot}'].count()
            boot_accuracies[boot, :] = accuracy_boot.reset_index()[layer + f'_correct{boot}']

        accuracy = np.mean(boot_accuracies, axis=0)
        std_accuracy = np.std(boot_accuracies, axis=0)
        accuracies[layer] = accuracy
        std_accuracies[layer] = std_accuracy


    # Accuracy per mask plot
    fig, ax =  plt.subplots()
    colours = ['red', 'blue', 'orange', 'green', 'yellow', 'purple']
    for i in range(len(xlabels)):
        layer = xlabels[i]
        accuracy = accuracies[layer]
        std = std_accuracies[layer]
        accuracy_df = pd.DataFrame()
        accuracy_df['mask_type'] = pd.unique(select_trial_df['mask_type'])
        accuracy_df['mean'] = accuracy
        accuracy_df['std'] = std
        accuracy_df = accuracy_df.reset_index()
        sns.lineplot(data=accuracy_df, x='mask_type', y='mean', color=colours[i])
        plt.fill_between(np.arange(n_masks), accuracy_df['mean'] - accuracy_df['std'], accuracy_df['mean'] + accuracy_df['std'], color= colours[i], alpha=0.2)
    
    plt.figlegend(handles = [
        Line2D([], [], marker='_', color="red", label="Layer 1"),
        Line2D([], [], marker='_', color="blue", label="Layer 2"),
        Line2D([], [], marker='_', color="orange", label="Layer 3"),
        Line2D([], [], marker='_', color="green", label="Layer 4"),
        Line2D([], [], marker='_', color="yellow", label="Layer 5"),
        Line2D([], [], marker='_', color="purple", label="All layers")])
    
    plt.ylabel("Accuracy")
    plt.xlabel("Layer")
    ax.set_xticklabels(['natural', 'scrambled', 'geometric', 'lines', 'blocked'])
    plt.title("Accuracy for different maks type")
    fig.tight_layout()
    file_name = os.path.join(wd, 'analysis/Accuracies-per-mask.png')
    fig.savefig(file_name)  


    # Calc C (criterion) for different masks accross splits
    print("Calculating C across folds")
    Z = stats.norm.ppf
    fold_count = 0
    fig, ax = plt.subplots()
    for cv_train_ind, cv_test_ind in cv.split(X_train, y_train):
        og_train_ind = out_train_inds[cv_train_ind]
        fold_data = data.iloc[og_train_ind]
        hit_rate_adjusted = ((fold_data[fold_data['valid_cue'] == 1].groupby(['mask_type', 'subject_nr']).sum()['answer'])+ 0.5) /((fold_data[fold_data['valid_cue'] == 1].groupby(['mask_type', 'subject_nr']).count()['valid_cue'] ) + 1)
        fa_rate_adjusted = ((fold_data[fold_data['valid_cue'] == 0].groupby(['mask_type', 'subject_nr']).sum()['answer'] ) + 0.5) / ((fold_data[fold_data['valid_cue'] == 0].groupby(['mask_type', 'subject_nr']).count()['valid_cue'] ) + 1)
        C_adjusted = -0.5*(hit_rate_adjusted.apply(lambda x: Z(x)) + fa_rate_adjusted.apply(lambda x: Z(x)))

        sns.lineplot(data=C_adjusted.reset_index(), x='mask_type', y=0)
        sns.despine(offset=2, trim=True)
        plt.tight_layout()

        fold_count += 0

    plt.ylabel("C")
    plt.title("Average loglinear C per mask across folds")
    fig.tight_layout()
    file_name = os.path.join(wd, 'analysis/C_across_folds.png')
    fig.savefig(file_name)    


    # Recursive feature elimination
    # Set up pipeline
    scaler = preprocessing.StandardScaler()
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y=y)
    lr_basemodel = LogisticRegression(max_iter=5000, class_weight = {0:class_weights[0], 1:class_weights[1]})

    pipe= MyPipeline(steps=[('pre', scaler),
                    ('lr', lr_basemodel)])
    fig, ax = plt.subplots()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    visualizer = RFECV(pipe, cv=cv, scoring='f1', step=0.1, ax=ax)
    visualizer.fit(X_train, y_train) 
    visualizer.finalize()
    ax.set_ylabel("F1 score")
    filename = os.path.join(wd, 'analysis/rfecv.png')
    fig.savefig(filename)

    # Save fitted models and selected features
    # selected_features = np.where(np.array(visualizer.support_) == 1)
    selected_features = visualizer.support_
    best_score = np.max(np.mean(visualizer.cv_scores_, axis=1))
    fitted_model = visualizer.rfe_estimator_.estimator_.named_steps['lr']


    file_name = os.path.join(wd, "analysis/rfecv_selected.npy")
    np.save(file_name, selected_features)
    file_name = os.path.join(wd, "analysis/rfecv_fit.pkl")
    pkl.dump(fitted_model, open(filename, "wb"))
    
    # Load fitted model
    file_name = os.path.join(wd, "analysis/rfecv_fit.pkl")
    fitted_model = pkl.load(open(os.path.join(wd, file_name),'rb'))

    # Hyper parameter tuning
    param_grid= {'lr__C': [0.1, 0.5, 1, 5, 10, 50, 100]}
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    scoring = ['accuracy', 'f1', 'precision', 'recall']
    gs = GridSearchCV(
        estimator=fitted_model,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1",
        n_jobs=-1,
        cv=cv
    )

    gs.fit(X_train, y_train)
    results = gs.cv_results_
    best_fit = gs.best_estimator_
    best_score = gs.best_score_
    print(f"Best fit {best_fit}")
    print(f"Best F1: {best_score}")

    # Save best fit
    file_name = os.path.join(wd, "analysis/gs_bestfit.pkl")
    pkl.dump(fitted_model, open(file_name, "wb"))


    # Calculate noise ceiling --> evaluation test data?
    participants = pd.unique(data['subject_nr'])
    nr_participants = len(participants)
    noiseLower = np.zeros(nr_participants)
    noiseHigher = np.zeros(nr_participants)
    
    GA_response = trial_df.groupby(['index'])['response'].mean()

    for i in range(nr_participants):
        sub = participants[i]
        sub_data = trial_df[trial_df['subject_nr']==sub]
        sub_mean_response = sub_data.groupby(['index'])['response'].mean()
        noiseHigher[i] = stats.spearmanr(sub_mean_response, GA_response)[0]

        selection = np.ones(nr_participants,dtype=bool)
        selection[i] = 0
        subs_without = participants[selection]
        sub_without_data = trial_df[trial_df['subject_nr'].isin(subs_without)]
        GA_without_response = sub_without_data.groupby(['index'])['response'].mean()
        noiseLower[i] = stats.spearmanr(sub_mean_response, GA_without_response)[0]

    noiseCeiling = {}
    noiseCeiling['UpperBound'] = np.mean(noiseHigher, axis=0)
    noiseCeiling['LowerBound'] = np.mean(noiseLower, axis=0)

    # Relate predictions back to noise ceiling
    predictions = logit_model.predict(X_scaled)
    prediction_probs = logit_model.predict_proba(X_scaled)
    # trial_df['prediction'] = predictions
    # model_mean_response = trial_df.groupby(['index'])['prediction'].mean()
    report_chance = prediction_probs[:, 1]
    trial_df['report_chance'] = report_chance
    mean_report_chance = trial_df.groupby(['index'])['report_chance'].mean()
    model_cor = stats.spearmanr(mean_report_chance, GA_response)[0]

    # Alternative; more stats info
    logit_model1 = sm.Logit(y, X_scaled) #inversion error
    results = logit_model1.fit()
    results.summary()

def PLS_model1(data, exp_features, random_features):
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    from sklearn.model_selection import RepeatedKFold, StratifiedKFold, Kfold, GridSearchCV, cross_validate, train_test_split, cross_val_predict, GroupShuffleSplit
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.pipeline import make_pipeline, Pipeline
    import statsmodels.api as sm
    from sklearn import preprocessing
    from yellowbrick.model_selection import RFECV
    from sklearn.cross_decomposition import PLSRegression

    file_dir = os.path.join(wd, 'analysis', 'image_paths_exp.csv') # should be cc1
    image_paths = pd.read_csv(file_dir)['path'].tolist()
    concepts = pd.unique(concept_selection['concept']).tolist()

    n_trials = len(trial_file)
    trial_df = pd.DataFrame()
   
    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']
    n_layers = len(layers)
    no_mask = [i for i in range(len(trial_file)) if trial_file.iloc[i]['mask_type']=='no_mask']
    n_mask_trials = n_trials - len(no_mask)

    print("Creating trial df")
    trial_df = pd.DataFrame()
    valid_data =  data[data['valid_cue'] == 1]

    no_mask_id = 0
    mask_trials = []

    all_index = []
    all_responses = []
    all_mask_types = []
    all_mask_activations = []
    all_target_activations = []
    all_rmask_activations = []
    all_rtarget_activations = []
    all_no_mask_ids = []

    for i in range(len(trial_file)):

        trial = trial_file.iloc[i]
        trial_id  = i

        # Check if trial is mask trial
        mask_path = trial['mask_path']

        if mask_path != 'no_mask':
            mask_trials.append(i)
            
            # get target path
            target_path = trial['ImageID']
            target_path = os.path.join(wd, target_path[53:])

            # get according activations
            target_index = image_paths.index(target_path)
            target_activations  = np.zeros((n_components, n_layers))
            rtarget_activations  = np.zeros((n_components, n_layers))
            for i in range(n_layers):
                layer = layers[i]
                target_activation = exp_features[layer][target_index, :]
                target_activations[:, i] = target_activation

                rtarget_activation = random_features[layer][target_index, :]
                rtarget_activations[:, i] = rtarget_activation
            
            all_target_activations.append(target_activations)
            all_rtarget_activations.append(target_activations)
                        
            # get mask path
            mask_path = os.path.join(wd, mask_path[53:])

            # get according activation
            mask_index = image_paths.index(mask_path)
            mask_activations  = np.zeros((n_components, n_layers))
            rmask_activations  = np.zeros((n_components, n_layers))
            for i in range(n_layers):
                layer = layers[i]
                mask_activation = exp_features[layer][mask_index, :]
                mask_activations[:, i] = mask_activation

                rmask_activation = random_features[layer][mask_index, :]
                rmask_activations[:, i] = rmask_activation
            
            all_mask_activations.append(target_activations)
            all_rmask_activations.append(target_activations)

            # get response for only valid trials 
            response = np.mean(valid_data[valid_data['index'] == trial_id]['answer'].tolist()) #anwer or correct?
            mask_type = mask_path.split('/')[-3]
            if mask_type == 'masks':
                mask_type = mask_path.split('/')[-2]

            all_responses.append(response)
            all_mask_types.append(mask_type)
            all_index.append(trial_id)
            all_no_mask_ids.append(no_mask_id)

            no_mask_id =+ 1

    trial_df['index'] = all_index
    trial_df['no_mask_id'] = all_no_mask_ids
    trial_df['mask_type'] = all_mask_types
    trial_df['response'] = all_responses
    trial_df['target_activation'] = all_target_activations
    trial_df['mask_activation'] = all_mask_activations
    trial_df['rtarget_activation'] = all_rtarget_activations
    trial_df['rmask_activation'] = all_rmask_activations

    # Activations for all trials, all ppn --> without PCA? 
    X1 = np.zeros((len(trial_df), n_components, n_layers))
    X2 = np.zeros((len(trial_df), n_components, n_layers))

    rX1 =  np.zeros((len(trial_df), n_components, n_layers))
    rX2 = np.zeros((len(trial_df), n_components, n_layers))

    for i in range(len(trial_df)):
            X1[i, :, :] = trial_df['target_activation'].iloc[i]
            X2[i, :, :] = trial_df['mask_activation'].iloc[i]
            rX1[i, :, :] = trial_df['rtarget_activation'].iloc[i]
            rX2[i, :, :] = trial_df['rmask_activation'].iloc[i]
    X = np.concatenate((X1, X2), axis=1)
    rX = np.concatenate((rX1, rX2), axis=1)
    y = np.asarray(trial_df['response'])

    # Split in development (train) and test / balanced for mask type
    indices = np.arange(len(trial_df))
    X_train, X_test, y_train, y_test, out_train_inds, out_test_inds = train_test_split(X, y, indices, test_size=0.3, stratify=trial_df['mask_type'].values, random_state=0)

    X_train = X[out_train_inds]
    X_test = X[out_test_inds]
    rX_train = rX[out_train_inds]
    rX_test = rX[out_test_inds]
    y_train = y[out_train_inds]
    y_test = y[out_test_inds]

    shell()

    # Simple model (without feature eliminatinon / grid search) + CV
    pls_basemodel = PLSRegression(n_components=2, max_iter=5000) # 500 components? 

    clf = make_pipeline(preprocessing.StandardScaler(), pls_basemodel)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    all_scores = []
    all_pred = []
    all_pred_ind = []
    all_random_scores = []
    for i in range(n_layers):
        print(f"{layers[i]}")

        # Imagnet Network Activations - LR
        # scores = cross_validate(clf, X_train[:, :, i], y_train, scoring=scoring, cv=cv, return_train_score=True)
        # # scores = cross_validate(clf, X_train[:, :, i], y_train, scoring=scoring, cv=cv)
        # all_scores.append(scores)
        # mean_accuracy = np.mean(scores['test_accuracy'])
        # mean_f1 = np.mean(scores['test_f1'])
        # mean_recall = np.mean(scores['test_recall'])
        # mean_precision = np.mean(scores['test_precision'])
        # print(f"Mean accuracy: {mean_accuracy}, std {np.std(scores['test_accuracy'])}") #0.555
        # print(f"Mean f1: {mean_f1}, std {np.std(scores['test_f1'])}") #0.557
        # print(f"Mean recall: {mean_recall}, std {np.std(scores['test_recall'])} ") #0.476
        # print(f"Mean precision: {mean_precision}, std {np.std(scores['test_precision'])}") #0.672

        # Cross validate prediction
        print(f"Cross val prediction")
        boot_inds = []
        boot_res = np.zeros((10, X_train.shape[0]))
        for boot in range(10):
            print(f"Boot {boot}")
            boot_cv = KFold(n_splits=5, shuffle=True, random_state=boot)

            boot_test_ind = []
            for train_index, test_index in boot_cv.split(X_train, y_train):
                boot_test_ind.append(test_index.tolist())

            boot_test_ind = [x for boot in boot_test_ind for x in boot]
            boot_inds.append(boot_test_ind)

            y_train_pred = cross_val_predict(clf, X_train[:, :, i], y_train, cv=boot_cv)
            boot_res[boot, :] = y_train_pred.reshape(1,-1)
        all_pred.append(np.mean(boot_res, axis=0))
        all_pred_ind.append(boot_inds)

        # Random network activations - LR
        # random_scores = cross_validate(clf, rX_train[:, :, i], y_train, scoring=scoring, cv=cv, return_train_score=True)
        # all_random_scores.append(random_scores)
        # mean_accuracy = np.mean(random_scores['test_accuracy'])
        # mean_f1 = np.mean(random_scores['test_f1'])
        # mean_recall = np.mean(random_scores['test_recall'])
        # mean_precision = np.mean(random_scores['test_precision'])
        # print(f"Mean accuracy random: {mean_accuracy}, std {np.std(random_scores['test_accuracy'])}") #0.555
        # print(f"Mean f1 random: {mean_f1}, std {np.std(random_scores['test_f1'])}") #0.557
        # print(f"Mean recall random: {mean_recall}, std {np.std(random_scores['test_recall'])} ") #0.476
        # print(f"Mean precision random: {mean_precision}, std {np.std(random_scores['test_precision'])}") #0.672
    
    shell()

def pls_da(X_train, y_train, X_test):
    from sklearn.cross_decomposition import PLSRegression

    # Define the PLS object for binary classification
    plsda = PLSRegression(n_components=500)
    
    # Fit the training set
    plsda.fit(X_train, y_train)
    
    # Binary prediction on the test set, done with thresholding
    binary_prediction = (plsda.predict(X_test)[:,0] > 0.5).astype('uint8')
    
    return binary_prediction

def PLS_model(data, exp_features, random_features):
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    from sklearn.model_selection import RepeatedKFold, StratifiedKFold, GridSearchCV, cross_validate, train_test_split, cross_val_predict, GroupShuffleSplit
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.pipeline import make_pipeline, Pipeline
    import statsmodels.api as sm
    from sklearn import preprocessing
    from yellowbrick.model_selection import RFECV
    from sklearn.cross_decomposition import PLSRegression

    # Extract features
    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']
    return_nodes = {
        'relu': 'layer1',
        'layer1.2.relu': 'layer2', 
        'layer2.3.relu': 'layer3',
        'layer3.5.relu': 'layer4', 
        'layer4.2.relu': 'layer5'
    }
    
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    # Preprocess all experimental images to fit PCA
    n_images = len(all_images)
    imgs = np.zeros((n_images, 3, 224, 224))
    new_paths = []
    for i in range(n_images):
        if i % 100 == 0:
            print(f"Preprocessing image {i}")
        img_path = all_images[i]
        img_path = os.path.join(wd, img_path[53:])
        new_paths.append(img_path)
        img = np.asarray(Image.open(img_path))
        img = transform(img)
        img = img.reshape(1, 3, 224, 224)
        imgs[i, :, :, :] = img

    # Extract features pretrained model
    print(f"Extracting features")
    imgs = torch.from_numpy(imgs).type(torch.DoubleTensor)
    exp_features = feature_extractor(imgs) 

    # Extract features random model
    print(f"Extracting features")
    random_model = models.resnet50(pretrained=False)
    feature_extractor = create_feature_extractor(random_model, return_nodes=return_nodes)
    random_features = feature_extractor(imgs) 

    # Predine arrays
    X1_single = {}
    rX1_single = {}
    X2_single = {}
    rX2_single = {}
    all_activations =[X1_single, rX1_single, X2_single, rX2_single]

    print("Creating trial df")
    file_dir = os.path.join(wd, 'analysis', 'image_paths_exp.csv') # should be cc1
    image_paths = pd.read_csv(file_dir)['path'].tolist()
    concepts = pd.unique(concept_selection['concept']).tolist()

    n_trials = len(trial_file)
    trial_df = pd.DataFrame()
   
    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']
    n_layers = len(layers)
    no_mask = [i for i in range(len(trial_file)) if trial_file.iloc[i]['mask_type']=='no_mask']
    n_mask_trials = n_trials - len(no_mask)

    no_mask_id = 0
    mask_trials = []
    for i in range(len(trial_file)):

        tmp = pd.DataFrame()
        # tmp_single = pd.DataFrame()

        trial = trial_file.iloc[i]
        trial_id  = i

        # Check if trial is mask trial
        mask_path = trial['mask_path']

        if mask_path != 'no_mask':
            mask_trials.append(i)
            
            # get target path
            target_path = trial['ImageID']
            target_path = os.path.join(wd, target_path[53:])

            # get according activations
            target_index = image_paths.index(target_path)
            target_activations  = np.zeros((n_components, n_layers))
            rtarget_activations  = np.zeros((n_components, n_layers))
            for i in range(n_layers):
                layer = layers[i]
                target_activation = exp_features[layer][target_index, :]
                target_activations[:, i] = target_activation

                rtarget_activation = random_features[layer][target_index, :]
                rtarget_activations[:, i] = rtarget_activation
                        
            # get mask path
            mask_path = os.path.join(wd, mask_path[53:])

            # get according activation
            mask_index = image_paths.index(mask_path)
            mask_activations  = np.zeros((n_components, n_layers))
            rmask_activations  = np.zeros((n_components, n_layers))
            for i in range(n_layers):
                layer = layers[i]
                mask_activation = exp_features[layer][mask_index, :]
                mask_activations[:, i] = mask_activation

                rmask_activation = random_features[layer][mask_index, :]
                rmask_activations[:, i] = rmask_activation

            # get response for all participants (average?)
            responses = data[data['index'] == trial_id]['answer'].tolist() #anwer or correct?
            subject_nrs = data[data['index'] == trial_id]['subject_nr'].tolist()
            valid = data[data['index'] == trial_id]['valid_cue'].tolist()
            mask_type = mask_path.split('/')[-3]
            if mask_type == 'masks':
                mask_type = mask_path.split('/')[-2]
            
            tmp['index'] = [trial_id for i in range(len(responses))]
            tmp['response'] = responses
            tmp['valid'] = valid
            tmp['subject_nr'] = subject_nrs
            tmp['target_path'] = [target_path for i in range(len(responses))]
            tmp['mask_path'] = [mask_path for i in range(len(responses))]
            tmp['mask_type'] = [mask_type for i in range(len(responses))]
            tmp['mask_activation'] = [mask_activations for i in range(len(responses))]
            tmp['target_activation'] = [target_activations for i in range(len(responses))]
            tmp['rmask_activation'] = [rmask_activations for i in range(len(responses))]
            tmp['rtarget_activation'] = [rtarget_activations for i in range(len(responses))]

            trial_df = pd.concat([trial_df, tmp], ignore_index=True)

            no_mask_id =+ 1

    # Only get valid trials 
    select_trial_df = trial_df[trial_df['valid']==1].reset_index()
    
    # Activations for all trials, all ppn
    print("Splitting datasets")
    X1 = np.zeros((len(select_trial_df), n_components, n_layers))
    X2 = np.zeros((len(select_trial_df), n_components, n_layers))

    rX1 =  np.zeros((len(select_trial_df), n_components, n_layers))
    rX2 = np.zeros((len(select_trial_df), n_components, n_layers))

    for i in range(len(select_trial_df)):
            X1[i, :, :] = select_trial_df['target_activation'].iloc[i]
            X2[i, :, :] = select_trial_df['mask_activation'].iloc[i]
            rX1[i, :, :] = select_trial_df['rtarget_activation'].iloc[i]
            rX2[i, :, :] = select_trial_df['rmask_activation'].iloc[i]
    X = np.concatenate((X1, X2), axis=1)
    rX = np.concatenate((rX1, rX2), axis=1)
    y = np.asarray(select_trial_df['response'])

    # Split in development (train) and test / balanced for mask type
    indices = np.arange(len(select_trial_df))
    X_train, X_test, y_train, y_test, out_train_inds, out_test_inds = train_test_split(X, y, indices, test_size=0.1, stratify=select_trial_df['mask_type'].values, random_state=0)

    X_dev = X[out_train_inds]
    X_test = X[out_test_inds]
    rX_dev = rX[out_train_inds]
    rX_test = rX[out_test_inds]
    y_dev = y[out_train_inds]
    y_test = y[out_test_inds]

    # All layers in one predictor
    all_X1 = np.zeros((len(select_trial_df), n_components * n_layers))
    all_X2 = np.zeros((len(select_trial_df), n_components * n_layers))

    all_rX1 = np.zeros((len(select_trial_df), n_components * n_layers))
    all_rX2 = np.zeros((len(select_trial_df), n_components * n_layers))

    for i in range(len(select_trial_df)):
        for layer in range(n_layers):
            all_X1[i, layer * n_components : (layer * n_components) + n_components] = select_trial_df['target_activation'].iloc[i][:, layer]
            all_X2[i, layer * n_components : (layer * n_components) + n_components] = select_trial_df['mask_activation'].iloc[i][:, layer]
            all_rX1[i, layer * n_components : (layer * n_components) + n_components] = select_trial_df['rtarget_activation'].iloc[i][:, layer]
            all_rX2[i, layer * n_components : (layer * n_components) + n_components] = select_trial_df['rmask_activation'].iloc[i][:, layer]

    all_X = np.concatenate((all_X1, all_X2), axis=1)
    all_rX = np.concatenate((all_rX1, all_rX2), axis=1)

    # Split in development (train) and test / balanced for mask type
    all_X_dev = all_X[out_train_inds]
    all_X_test = all_X[out_test_inds]
    all_rX_dev = all_rX[out_train_inds]
    all_rX_test = all_rX[out_test_inds]

    shell()

    # -------------------------------------------------- PLS model
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score

    print("Start bootstrap")
    layers_oi = layers + ['all']
    n_layers = len(layers_oi)
    boots = 10

    all_train_accuracies = np.zeros((boots, n_layers))
    all_train_f1s = np.zeros((boots, n_layers))
    all_train_recalls = np.zeros((boots, n_layers))
    all_train_precisions = np.zeros((boots, n_layers))
    all_train_r_accuracies = np.zeros((boots, n_layers))
    all_train_r_f1s = np.zeros((boots, n_layers))
    all_train_r_recalls = np.zeros((boots, n_layers))
    all_train_r_precisions = np.zeros((boots, n_layers))

    all_test_accuracies = np.zeros((boots, n_layers))
    all_test_f1s = np.zeros((boots, n_layers))
    all_test_recalls = np.zeros((boots, n_layers))
    all_test_precisions = np.zeros((boots, n_layers))
    all_test_r_accuracies = np.zeros((boots, n_layers))
    all_test_r_f1s = np.zeros((boots, n_layers))
    all_test_r_recalls = np.zeros((boots, n_layers))
    all_test_r_precisions = np.zeros((boots, n_layers))

    boots_test_inds = []
    boots_predictions = []

    for boot in range(boots):

        print(f"boot {boot}")
        seed = boot
        boot_test_inds = []
        boot_prediction = {}

        layers_train_accuracies = []
        layers_train_f1s = []
        layers_train_recalls = []
        layers_train_precisions = []
        layers_train_r_accuracies = []
        layers_train_r_f1s = []
        layers_train_r_recalls = []
        layers_train_r_precisions = []

        layers_test_accuracies = []
        layers_test_f1s = []
        layers_test_recalls = []
        layers_test_precisions = []
        layers_test_r_accuracies = []
        layers_test_r_f1s = []
        layers_test_r_recalls = []
        layers_test_r_precisions = []

        cval = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        for i in range(n_layers):
            layer = layers_oi[i]
            print(f"{layer}")

            train_accuracies = []
            train_f1s = []
            train_recalls = []
            train_precisions = []
            train_r_accuracies = []
            train_r_f1s = []
            train_r_recalls = []
            train_r_precisions = []

            test_accuracies = []
            test_f1s = []
            test_recalls = []
            test_precisions = []
            test_r_accuracies = []
            test_r_f1s = []
            test_r_recalls = []
            test_r_precisions = []

            split_count = 0

            predictions = []

            if layer != 'all':
                for train, test in cval.split(X_dev, y_dev):
                    print(f"split : {split_count}")

                    # Scale features
                    scaler = preprocessing.StandardScaler()
                    sX_dev_train = scaler.fit_transform(X_dev[train,:,i])
                    sX_dev_test = scaler.transform(X_dev[test,:,i])

                    rscaler = preprocessing.StandardScaler()
                    rsX_dev_train = rscaler.fit_transform(rX_dev[train,:,i])
                    rsX_dev_test = rscaler.transform(rX_dev[test,:,i])

                    # Predict
                    test_y_pred = pls_da(sX_dev_train, y_dev[train], sX_dev_test)  
                    test_ry_pred = pls_da(rsX_dev_train, y_dev[train], rsX_dev_test)  
                    train_y_pred = pls_da(sX_dev_train, y_dev[train], sX_dev_train)  
                    train_ry_pred = pls_da(rsX_dev_train, y_dev[train], rsX_dev_train) 

                    # Performance - test set
                    test_accuracies.append(accuracy_score(y_dev[test], test_y_pred))
                    test_f1s.append(f1_score(y_dev[test], test_y_pred))
                    test_recalls.append(recall_score(y_dev[test], test_y_pred))
                    test_precisions.append(precision_score(y_dev[test], test_y_pred))

                    test_r_accuracies.append(accuracy_score(y_dev[test], test_ry_pred))
                    test_r_f1s.append(f1_score(y_dev[test], test_ry_pred))
                    test_r_recalls.append(recall_score(y_dev[test], test_ry_pred))
                    test_r_precisions.append(precision_score(y_dev[test], test_ry_pred))

                    # Performance - train set 
                    train_accuracies.append(accuracy_score(y_dev[train], train_y_pred))
                    train_f1s.append(f1_score(y_dev[train], train_y_pred))
                    train_recalls.append(recall_score(y_dev[train], train_y_pred))
                    train_precisions.append(precision_score(y_dev[train], train_y_pred))

                    train_r_accuracies.append(accuracy_score(y_dev[train], train_ry_pred))
                    train_r_f1s.append(f1_score(y_dev[train], train_ry_pred))
                    train_r_recalls.append(recall_score(y_dev[train], train_ry_pred))
                    train_r_precisions.append(precision_score(y_dev[train], train_ry_pred))
        
                    for i in range(len(test)):
                        boot_test_inds.append(test[i])
                        predictions.append(test_y_pred[i])

                    split_count +=1     
            else:
                for train, test in cval.split(X_dev, y_dev):
                    print(f"split : {split_count}")

                    # Scale features
                    scaler = preprocessing.StandardScaler()
                    sX_dev_train = scaler.fit_transform(all_X_dev[train, :])
                    sX_dev_test = scaler.transform(all_X_dev[test, :])

                    rscaler = preprocessing.StandardScaler()
                    rsX_dev_train = rscaler.fit_transform(all_rX_dev[train, :])
                    rsX_dev_test = rscaler.transform(all_rX_dev[test, :])

                    # Predict
                    test_y_pred = pls_da(sX_dev_train, y_dev[train], sX_dev_test)  
                    test_ry_pred = pls_da(rsX_dev_train, y_dev[train], rsX_dev_test)  
                    train_y_pred = pls_da(sX_dev_train, y_dev[train], sX_dev_train)  
                    train_ry_pred = pls_da(rsX_dev_train, y_dev[train], rsX_dev_train) 

                    # Performance - test set
                    test_accuracies.append(accuracy_score(y_dev[test], test_y_pred))
                    test_f1s.append(f1_score(y_dev[test], test_y_pred))
                    test_recalls.append(recall_score(y_dev[test], test_y_pred))
                    test_precisions.append(precision_score(y_dev[test], test_y_pred))

                    test_r_accuracies.append(accuracy_score(y_dev[test], test_ry_pred))
                    test_r_f1s.append(f1_score(y_dev[test], test_ry_pred))
                    test_r_recalls.append(recall_score(y_dev[test], test_ry_pred))
                    test_r_precisions.append(precision_score(y_dev[test], test_ry_pred))

                    # Performance - train set 
                    train_accuracies.append(accuracy_score(y_dev[train], train_y_pred))
                    train_f1s.append(f1_score(y_dev[train], train_y_pred))
                    train_recalls.append(recall_score(y_dev[train], train_y_pred))
                    train_precisions.append(precision_score(y_dev[train], train_y_pred))

                    train_r_accuracies.append(accuracy_score(y_dev[train], train_ry_pred))
                    train_r_f1s.append(f1_score(y_dev[train], train_ry_pred))
                    train_r_recalls.append(recall_score(y_dev[train], train_ry_pred))
                    train_r_precisions.append(precision_score(y_dev[train], train_ry_pred))

                    for i in range(len(test)):
                        boot_test_inds.append(test[i])
                        predictions.append(test_y_pred[i])

                    split_count +=1     

            boot_prediction[layer] = predictions

            layers_train_accuracies.append(np.mean(train_accuracies))
            layers_train_f1s.append(np.mean(train_f1s))
            layers_train_recalls.append(np.mean(train_recalls))
            layers_train_precisions.append(np.mean(train_precisions))
            layers_train_r_accuracies.append(np.mean(train_r_accuracies))
            layers_train_r_f1s.append(np.mean(train_r_f1s))
            layers_train_r_recalls.append(np.mean(train_r_recalls))
            layers_train_r_precisions.append(np.mean(train_r_precisions))

            layers_test_accuracies.append(np.mean(test_accuracies))
            layers_test_f1s.append(np.mean(test_f1s))
            layers_test_recalls.append(np.mean(test_recalls))
            layers_test_precisions.append(np.mean(test_precisions))
            layers_test_r_accuracies.append(np.mean(test_r_accuracies))
            layers_test_r_f1s.append(np.mean(test_r_f1s))
            layers_test_r_recalls.append(np.mean(test_r_recalls))
            layers_test_r_precisions.append(np.mean(test_r_precisions))

        boots_predictions.append(boot_prediction)

        all_train_accuracies[boot, :] = layers_train_accuracies
        all_train_f1s[boot, :] = layers_train_f1s
        all_train_recalls[boot, :] = layers_train_recalls
        all_train_precisions[boot, :] = layers_train_precisions
        all_train_r_accuracies[boot, :] = layers_train_r_accuracies
        all_train_r_f1s[boot, :] = layers_train_r_f1s
        all_train_r_recalls[boot, :] = layers_train_r_recalls
        all_train_r_precisions[boot, :] = layers_train_r_precisions

        all_test_accuracies[boot, :] = layers_test_accuracies
        all_test_f1s[boot, :] = layers_test_f1s
        all_test_recalls[boot, :] = layers_test_recalls
        all_test_precisions[boot, :] = layers_test_precisions
        all_test_r_accuracies[boot, :] = layers_test_r_accuracies
        all_test_r_f1s[boot, :] =  layers_test_r_f1s
        all_test_r_recalls[boot, :] = layers_test_r_recalls
        all_test_r_precisions[boot, :] = layers_test_r_precisions

    score_df = pd.DataFrame()
    score_df['layer'] = layers_oi
    score_df['m_test_accuracy'] = np.mean(all_test_accuracies, axis=0)
    score_df['m_test_f1'] = np.mean(all_test_f1s, axis=0)
    score_df['m_test_precision'] = np.mean(all_test_precisions, axis=0)
    score_df['m_test_recall'] = np.mean(all_test_recalls, axis=0)
    score_df['m_test_r_accuracy'] = np.mean(all_test_r_accuracies, axis=0)
    score_df['m_test_r_f1'] = np.mean(all_test_r_f1s, axis=0)
    score_df['m_test_r_precision'] = np.mean(all_test_r_precisions, axis=0)
    score_df['m_test_r_recall'] = np.mean(all_test_r_recalls, axis=0)
    score_df['m_train_accuracy'] = np.mean(all_train_accuracies, axis=0)
    score_df['m_train_f1'] = np.mean(all_train_f1s, axis=0)
    score_df['m_train_precision'] = np.mean(all_train_precisions, axis=0)
    score_df['m_train_recall'] = np.mean(all_train_recalls, axis=0)
    score_df['m_train_r_accuracy'] = np.mean(all_train_r_accuracies, axis=0)
    score_df['m_train_r_f1'] = np.mean(all_train_r_f1s, axis=0)
    score_df['m_train_r_precision'] = np.mean(all_train_r_precisions, axis=0)
    score_df['m_train_r_recall'] = np.mean(all_train_r_recalls, axis=0)
    score_df['s_test_accuracy'] = np.std(all_test_accuracies, axis=0)
    score_df['s_test_f1'] = np.std(all_test_f1s, axis=0)
    score_df['s_test_precision'] = np.std(all_test_precisions, axis=0)
    score_df['s_test_recall'] = np.std(all_test_recalls, axis=0)
    score_df['s_test_r_accuracy'] = np.std(all_test_r_accuracies, axis=0)
    score_df['s_test_r_f1'] = np.std(all_test_r_f1s, axis=0)
    score_df['s_test_r_precision'] = np.std(all_test_r_precisions, axis=0)
    score_df['s_test_r_recall'] = np.std(all_test_r_recalls, axis=0)
    score_df['s_train_accuracy'] = np.std(all_train_accuracies, axis=0)
    score_df['s_train_f1'] = np.std(all_train_f1s, axis=0)
    score_df['s_train_precision'] = np.std(all_train_precisions, axis=0)
    score_df['s_train_recall'] = np.std(all_train_recalls, axis=0)
    score_df['s_train_r_accuracy'] = np.std(all_train_r_accuracies, axis=0)
    score_df['s_train_r_f1'] = np.std(all_train_r_f1s, axis=0)
    score_df['s_train_r_precision'] = np.std(all_train_r_precisions, axis=0)
    score_df['s_train_r_recall'] = np.std(all_train_r_recalls, axis=0)

    #  Plot performance over layers
    n_masks = len(pd.unique(trial_df['mask_type']))

    fig, ax =  plt.subplots(2,2, sharex=True, dpi=100, figsize=(14,7))
    fig.suptitle('PLS model performance')
    fig.supxlabel('Layer')
    fig.supylabel('Score')
    sns.lineplot(data=score_df, x='layer', y='m_train_accuracy', color='red', linestyle='--', ax=ax[0,0])
    sns.lineplot(data=score_df, x='layer', y='m_test_accuracy', color='red', ax=ax[0,0])
    ax[0,0].fill_between(np.arange(n_layers), score_df['m_train_accuracy'] - score_df['s_train_accuracy'], score_df['m_train_accuracy'] + score_df['s_train_accuracy'], color= 'red', alpha=0.2)
    ax[0,0].fill_between(np.arange(n_layers), score_df['m_test_accuracy'] - score_df['s_test_accuracy'], score_df['m_test_accuracy'] + score_df['s_test_accuracy'], color= 'red', alpha=0.2)
    sns.lineplot(data=score_df, x='layer', y='m_train_r_accuracy', color='firebrick', linestyle='--', ax=ax[0,0])
    sns.lineplot(data=score_df, x='layer', y='m_test_r_accuracy', color='firebrick', ax=ax[0,0])
    ax[0,0].fill_between(np.arange(n_layers), score_df['m_train_r_accuracy'] - score_df['s_train_r_accuracy'], score_df['m_train_r_accuracy'] + score_df['s_train_r_accuracy'], color= 'firebrick', alpha=0.2)
    ax[0,0].fill_between(np.arange(n_layers), score_df['m_test_r_accuracy'] - score_df['s_test_r_accuracy'],  score_df['m_test_r_accuracy'] +  score_df['s_test_r_accuracy'], color= 'firebrick', alpha=0.2)
    ax[0,0].set_title("Accuracy")
    ax[0,0].yaxis.label.set_visible(False)
    ax[0,0].xaxis.label.set_visible(False)
    ax[0,0].legend(handles=[
        Line2D([], [], marker='_', color="red", label="Pretrained"), 
        Line2D([], [], marker='_', color="firebrick", label="Random")], loc='upper right')
    sns.lineplot(data=score_df, x='layer', y='m_train_f1', color='blue', linestyle='--', ax=ax[0,1])
    sns.lineplot(data=score_df, x='layer', y='m_test_f1', color='blue', ax=ax[0,1])
    ax[0,1].fill_between(np.arange(n_layers), score_df['m_train_f1'] - score_df['s_train_f1'], score_df['m_train_f1'] + score_df['s_train_f1'], color= 'blue', alpha=0.2)
    ax[0,1].fill_between(np.arange(n_layers), score_df['m_test_f1'] - score_df['s_test_f1'],  score_df['m_test_f1'] +  score_df['s_test_f1'], color= 'blue', alpha=0.2)
    sns.lineplot(data=score_df, x='layer', y='m_train_r_f1', color='darkblue', linestyle='--', ax=ax[0,1])
    sns.lineplot(data=score_df, x='layer', y='m_test_r_f1', color='darkblue', ax=ax[0,1])
    ax[0,1].fill_between(np.arange(n_layers), score_df['m_train_r_f1'] - score_df['s_train_r_f1'], score_df['m_train_r_f1'] + score_df['s_train_r_f1'], color= 'darkblue', alpha=0.2)
    ax[0,1].fill_between(np.arange(n_layers), score_df['m_test_r_f1'] - score_df['s_test_r_f1'],  score_df['m_test_r_f1'] +  score_df['s_test_r_f1'], color= 'darkblue', alpha=0.2)
    ax[0,1].set_title("F1")
    ax[0,1].yaxis.label.set_visible(False)
    ax[0,1].xaxis.label.set_visible(False)
    ax[0,1].legend(handles=[
        Line2D([], [], marker='_', color="blue", label="Pretrained"),
        Line2D([], [], marker='_', color="darkblue", label="Random")], loc='upper right')
    sns.lineplot(data=score_df, x='layer', y='m_train_precision', color='palegreen', linestyle='--', ax=ax[1,0])
    sns.lineplot(data=score_df, x='layer', y='m_test_precision', color='palegreen', ax=ax[1,0])
    ax[1,0].fill_between(np.arange(n_layers), score_df['m_train_precision'] - score_df['s_train_precision'], score_df['m_train_precision'] + score_df['s_train_precision'], color= 'palegreen', alpha=0.2)
    ax[1,0].fill_between(np.arange(n_layers), score_df['m_test_precision'] - score_df['s_test_precision'],  score_df['m_test_precision'] +  score_df['s_test_precision'], color= 'palegreen', alpha=0.2)
    sns.lineplot(data=score_df, x='layer', y='m_train_r_precision', color='darkgreen', linestyle='--', ax=ax[1,0])
    sns.lineplot(data=score_df, x='layer', y='m_test_r_precision', color='darkgreen', ax=ax[1,0])
    ax[1,0].fill_between(np.arange(n_layers), score_df['m_train_r_precision'] - score_df['s_train_r_precision'], score_df['m_train_r_precision'] + score_df['s_train_r_precision'], color= 'darkgreen', alpha=0.2)
    ax[1,0].fill_between(np.arange(n_layers), score_df['m_test_r_precision'] - score_df['s_test_r_precision'],  score_df['m_test_r_precision'] +  score_df['s_test_r_precision'], color= 'darkgreen', alpha=0.2)
    ax[1,0].set_title("Precision")
    ax[1,0].yaxis.label.set_visible(False)
    ax[1,0].xaxis.label.set_visible(False)
    ax[1,0].set_xticklabels(["1","2","3","4","5", "all"])
    ax[1,0].legend(handles=[
        Line2D([], [], marker='_', color="palegreen", label="Pretrained"),
        Line2D([], [], marker='_', color="darkgreen", label="Random")], loc='upper right')
    sns.lineplot(data=score_df, x='layer', y='m_train_recall', color='peachpuff', linestyle='--', ax=ax[1,1])
    sns.lineplot(data=score_df, x='layer', y='m_test_recall', color='peachpuff', ax=ax[1,1])
    ax[1,1].fill_between(np.arange(n_layers), score_df['m_train_recall'] - score_df['s_train_recall'], score_df['m_train_recall'] + score_df['s_train_recall'], color= 'peachpuff', alpha=0.2)
    ax[1,1].fill_between(np.arange(n_layers), score_df['m_test_recall'] - score_df['s_test_recall'],  score_df['m_test_recall'] +  score_df['s_test_recall'], color= 'peachpuff', alpha=0.2)
    sns.lineplot(data=score_df, x='layer', y='m_train_r_recall', color='darkorange', linestyle='--', ax=ax[1,1])
    sns.lineplot(data=score_df, x='layer', y='m_test_r_recall', color='darkorange', ax=ax[1,1])
    ax[1,1].fill_between(np.arange(n_layers), score_df['m_train_r_recall'] - score_df['s_train_r_recall'], score_df['m_train_r_recall'] + score_df['s_train_r_recall'], color= 'darkorange', alpha=0.2)
    ax[1,1].fill_between(np.arange(n_layers), score_df['m_test_r_recall'] - score_df['s_test_r_recall'],  score_df['m_test_r_recall'] +  score_df['s_test_r_recall'], color= 'darkorange', alpha=0.2)
    ax[1,1].set_title("Recall")
    ax[1,1].yaxis.label.set_visible(False)
    ax[1,1].xaxis.label.set_visible(False)
    ax[1,1].set_xticklabels(["1","2","3","4","5", "all"])
    ax[1,1].legend(handles=[        
        Line2D([], [], marker='_', color="peachpuff", label="Pretrained"),
        Line2D([], [], marker='_', color="darkorange", label="Random")], loc='upper right')

    plt.figlegend(handles = [
        Line2D([], [], marker='_', color="black", label="Test score"),
        Line2D([], [], marker='_', color="black", label="Training score", linestyle="--")
        ])

    fig.tight_layout()
    file_name = os.path.join(wd, 'analysis/TrainTest_Score_across_layers_PLS.png')
    fig.savefig(file_name)  




    # # Cross validate prediction
    # print(f"Cross val prediction")
    # boot_res = []
    # boot_inds = []
    # for boot in range(10):
    #     print(f"Boot {boot}")
    #     boot_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)

    #     boot_test_ind = []
    #     for train_index, test_index in boot_cv.split(X_train, y_train):
    #         # boot_test_inds.append(test_index)
    #         boot_test_ind.append(test_index.tolist())

    #     boot_test_ind = [x for boot in boot_test_ind for x in boot]
    #     boot_inds.append(boot_test_ind)

    #     y_train_pred = cross_val_predict(clf, X_train[:, :, i], y_train, cv=boot_cv)
    #     boot_res.append(y_train_pred)
    # all_pred.append(boot_res)
    # all_pred_ind.append(boot_inds)



    


def distance_analysis(data, exp_features, random_features):
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    from sklearn.model_selection import RepeatedKFold, StratifiedKFold, GridSearchCV, cross_validate, train_test_split, cross_val_predict, GroupShuffleSplit
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.pipeline import make_pipeline, Pipeline
    import statsmodels.api as sm
    from sklearn import preprocessing
    from yellowbrick.model_selection import RFECV
    from scipy import spatial

    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']
    return_nodes = {
        # node_name: user-specified key for output dict
        'relu': 'layer1',
        'layer1.2.relu': 'layer2', 
        'layer2.3.relu': 'layer3',
        'layer3.5.relu': 'layer4', 
        'layer4.2.relu': 'layer5'
    }
    
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    # Preprocess all experimental images to fit PCA
    n_images = len(all_images)
    imgs = np.zeros((n_images, 3, 224, 224))
    new_paths = []
    for i in range(n_images):
        if i % 100 == 0:
            print(f"Preprocessing image {i}")
        img_path = all_images[i]
        img_path = os.path.join(wd, img_path[53:])
        new_paths.append(img_path)
        img = np.asarray(Image.open(img_path))
        img = transform(img)
        img = img.reshape(1, 3, 224, 224)
        imgs[i, :, :, :] = img

    # Extract features pretrained model
    print(f"Extracting features")
    imgs = torch.from_numpy(imgs).type(torch.DoubleTensor)
    feature_dict = feature_extractor(imgs) 

    # Extract features random model
    print(f"Extracting features")
    random_model = models.resnet50(pretrained=False)
    feature_extractor = create_feature_extractor(random_model, return_nodes=return_nodes)
    random_feature_dict = feature_extractor(imgs) 

    file_dir = os.path.join(wd, 'analysis', 'image_paths_exp.csv') # should be cc1
    image_paths = pd.read_csv(file_dir)['path'].tolist()
    concepts = pd.unique(concept_selection['concept']).tolist()

    n_trials = len(trial_file)
    trial_df = pd.DataFrame()

    n_layers = len(layers)
    no_mask = [i for i in range(len(trial_file)) if trial_file.iloc[i]['mask_type']=='no_mask']
    n_mask_trials = n_trials - len(no_mask)

    print("Creating trial df")
    no_mask_id = 0

    X1_single = {}
    rX1_single = {}
    X2_single = {}
    rX2_single = {}
    all_activations =[X1_single, rX1_single, X2_single, rX2_single]

    # Predefine arrays
    for layer in layers:
        for activation in all_activations:
            activation[layer] = np.zeros((n_mask_trials, feature_dict[layer][0, :, :].flatten().shape[0]))

    for i in range(len(trial_file)):

        tmp = pd.DataFrame()
        # tmp_single = pd.DataFrame()

        trial = trial_file.iloc[i]
        trial_id  = i

        # Check if trial is mask trial
        mask_path = trial['mask_path']

        if mask_path != 'no_mask':
            
            # get target path
            target_path = trial['ImageID']
            target_path = os.path.join(wd, target_path[53:])

            # get according activations
            target_index = image_paths.index(target_path)

            # target_activations = []
            # rtarget_activations = []
            for layer in layers:
                target_activation = feature_dict[layer][target_index, :,:]
                target_activation = torch.flatten(target_activation).detach().numpy()
                X1_single[layer][no_mask_id, :] = target_activation

                rtarget_activation = random_feature_dict[layer][target_index, :,:]
                rtarget_activation = torch.flatten(rtarget_activation).detach().numpy()
                rX1_single[layer][no_mask_id, :] = rtarget_activation
    
            # get mask path
            mask_path = os.path.join(wd, mask_path[53:])

            # get according activation
            mask_index = image_paths.index(mask_path)

            # mask_activations = []
            # rmask_activations = []
            for layer in layers:
                mask_activation = feature_dict[layer][mask_index, :,:]
                mask_activation = torch.flatten(mask_activation).detach().numpy()
                X2_single[layer][no_mask_id, :] = mask_activation

                rmask_activation = feature_dict[layer][mask_index, :,:]
                rmask_activation = torch.flatten(rmask_activation).detach().numpy()
                rX2_single[layer][no_mask_id, :] = rmask_activation

            # get response for all participants (average?)
            responses = data[data['index'] == trial_id]['answer'].tolist() #anwer or correct?
            subject_nrs = data[data['index'] == trial_id]['subject_nr'].tolist()
            mask_type = mask_path.split('/')[-3]
            valid = data[data['index'] == trial_id]['valid_cue'].tolist()
            if mask_type == 'masks':
                mask_type = mask_path.split('/')[-2]

            tmp['index'] = [trial_id for i in range(len(responses))]
            tmp['response'] = responses
            tmp['valid'] = valid
            tmp['subject_nr'] = subject_nrs
            tmp['target_path'] = [target_path for i in range(len(responses))]
            tmp['mask_path'] = [mask_path for i in range(len(responses))]
            tmp['mask_type'] = [mask_type for i in range(len(responses))]
            tmp['no_mask_id'] = no_mask_id
            # tmp['mask_activation'] = [mask_activations for i in range(len(responses))]
            # tmp['target_activation'] = [target_activations for i in range(len(responses))]
            # tmp['rmask_activation'] = [rmask_activations for i in range(len(responses))]
            # tmp['rtarget_activation'] = [rtarget_activations for i in range(len(responses))]
            tmp['no_mask_id'] = [no_mask_id for i in range(len(responses))]

            trial_df = pd.concat([trial_df, tmp], ignore_index=True)

            no_mask_id += 1

    # Only get valid trials 
    select_trial_df = trial_df[trial_df['valid']==1].reset_index()

    X_distance = np.zeros((n_mask_trials, n_layers))
    rX_distance = np.zeros((n_mask_trials, n_layers))
    
    print("Calculating distances")
    # Calculate distances per trial and layer
    for i in range(n_mask_trials):
        for j in range(n_layers):
            layer = layers[j]
            target = X1_single[layer][i, :]
            mask = X2_single[layer][i, :]
            r_target = rX1_single[layer][i, :]
            r_mask = rX2_single[layer][i, :]

            distance = spatial.distance.correlation(target, mask)
            r_distance = spatial.distance.correlation(r_target, r_mask)

            X_distance[i, j] = distance
            rX_distance[i, j] = r_distance

    shell()

    # Add distances to select trial_df
    distance_df_out = pd.DataFrame()
    for i in range(n_mask_trials):
        distance_df_in = pd.DataFrame()
        for j in range(n_layers):

            layer = layers[j]

            tmp = pd.DataFrame()

            nr_trials = len(select_trial_df[select_trial_df['no_mask_id']==i])
            distance = X_distance[i, j]
            r_distance = rX_distance[i, j]
            distances = [distance for i in range(nr_trials)]
            r_distances = [r_distance for i in range(nr_trials)]
            tmp[layer + '_d'] = distances
            tmp[layer + 'r_d'] = r_distances
            distance_df_in = pd.concat([distance_df_in, tmp], axis=1)

        distance_df_out = pd.concat([distance_df_out, distance_df_in], ignore_index=True)

    select_trial_df = pd.concat([select_trial_df, distance_df_out], axis=1)

    # Plot spread of distances per mask type for pretrained and trained
    fig, ax =  plt.subplots(1,2, sharey=True)
    fig.suptitle('Mean target-mask distance per mask type')
    fig.supxlabel('Mask type')
    fig.supylabel('Pearson distance')
    colours = ['red', 'blue', 'orange', 'green', 'yellow']
    for i in range(len(layers)):
        layer = layers[i]
        sns.lineplot(data=select_trial_df, x='mask_type', y=f'{layer}_d', color=colours[i], ax=ax[0])
        sns.lineplot(data=select_trial_df, x='mask_type', y=f'{layer}r_d', color=colours[i], ax=ax[1])
    
    plt.figlegend(handles = [
        Line2D([], [], marker='_', color="red", label="Layer 1"),
        Line2D([], [], marker='_', color="blue", label="Layer 2"),
        Line2D([], [], marker='_', color="orange", label="Layer 3"),
        Line2D([], [], marker='_', color="green", label="Layer 4"),
        Line2D([], [], marker='_', color="yellow", label="Layer 5")])

    ax[0].set_xticklabels(['natural', 'scrambled', 'geometric', 'lines', 'blocked'])
    ax[1].set_xticklabels(['natural', 'scrambled', 'geometric', 'lines', 'blocked'])
    ax[0].xaxis.label.set_visible(False)
    ax[0].yaxis.label.set_visible(False)
    ax[1].xaxis.label.set_visible(False)
    ax[0].set_title("Pretrained activations")
    ax[1].set_title("Random activations")
    fig.tight_layout()
    file_name = os.path.join(wd, 'analysis/Distance-per-mask.png')
    fig.savefig(file_name) 

    # Alternative
    alt_df = pd.DataFrame()
    cors = []
    r_cors = []
    mask_types = []
    layer_names = []
    for layer in layers:
        # for i in range(len(select_trial_df)):
        cors.append(select_trial_df[layer+ '_d'].to_list())
        r_cors.append(select_trial_df[layer+'r_d'].to_list())
        mask_types.append(select_trial_df['mask_type'].to_list())
        layer= [layer for i in range(len(select_trial_df))]
        layer_names.append(layer)

    cors = [x for cor in cors for x in cor]
    r_cors = [x for r_cor in r_cors for x in r_cor]
    mask_types = [x for mask_type in mask_types for x in mask_type]
    layer_names = [x for layer in layer_names for x in layer]


    alt_df['r_distance'] = r_cors
    alt_df['distance'] = cors
    alt_df['mask_type'] = mask_types
    alt_df['layer'] = layer_names

    mask_types = pd.unique(select_trial_df['mask_type'])
    fig, ax =  plt.subplots(1,2, sharey=True)
    fig.suptitle('Mean target-mask distance per mask type')
    fig.supxlabel('Layer')
    fig.supylabel('Pearson distance')
    colours = ['red', 'blue', 'orange', 'green', 'yellow']
    for i in range(len(mask_types)):
        # layer = layers[i]
        mask_type = mask_types[i]
        sns.lineplot(data=alt_df[alt_df['mask_type']==mask_type], x='layer', y='distance', color=colours[i], ax=ax[0])
        sns.lineplot(data=alt_df[alt_df['mask_type']==mask_type], x='layer', y=f'r_distance', color=colours[i], ax=ax[1])
    
    plt.figlegend(handles = [
        Line2D([], [], marker='_', color="red", label="Natural"),
        Line2D([], [], marker='_', color="blue", label="Scrambled"),
        Line2D([], [], marker='_', color="orange", label="Geometric"),
        Line2D([], [], marker='_', color="green", label="Lines"),
        Line2D([], [], marker='_', color="yellow", label="Blocked")])

    ax[0].set_xticklabels([1,2,3,4,5])
    ax[0].xaxis.label.set_visible(False)
    ax[0].yaxis.label.set_visible(False)
    ax[1].xaxis.label.set_visible(False)
    ax[0].set_title("Pretrained activations")
    ax[1].set_title("Random activations")
    fig.tight_layout()
    file_name = os.path.join(wd, 'analysis/Distance-per-mask1.png')
    fig.savefig(file_name)  

    # Calc cor between response and distance  
    cors = []
    r_cors = []
    p_cors = []
    p_r_cors = []
    for j in range(n_layers):
        layer = layers[j]
        cor = stats.pearsonr(select_trial_df['response'], select_trial_df[layer + '_d'])
        cors.append(cor[0])
        p_cors.append(cor[1])
        r_cor = stats.pearsonr(select_trial_df['response'], select_trial_df[layer + 'r_d'])
        r_cors.append(r_cor[0])
        p_r_cors.append(r_cor[1])
    
    cor_df = pd.DataFrame()
    cor_df['cor'] = cors
    cor_df['r_cor'] = r_cors
    cor_df['p_cors'] = p_cors
    cor_df['p_r_cors'] = p_r_cors
    cor_df['layer'] = layers

    # Plot
    fig, ax = plt.subplots()
    sns.stripplot(x="layer", y="cor", data=cor_df, color='blue')
    sns.stripplot(x="layer", y="r_cor", data=cor_df, color='red')
    plt.ylabel("Correlation")
    plt.xlabel("Layer")
    ax.set_xticklabels([1,2,3,4,5])
    plt.title("Correlation target-mask distance and report")
    plt.figlegend(handles = [
        Line2D([], [], marker='o', color="red", label="Random activations", linestyle=""),
        Line2D([], [], marker='o', color="blue", label="Pretrained activations", linestyle="")])
    fig.tight_layout()
    file_name = os.path.join(wd, 'analysis/Distance-Report-cor.png')
    fig.savefig(file_name) 


    # Set up model
    X = np.zeros((len(select_trial_df), n_layers))
    X_random = np.zeros((len(select_trial_df), n_layers))

    for i in range(len(select_trial_df)): # check for other logit model (non -> scaled activatation)
            no_mask_id = select_trial_df.iloc[i]['no_mask_id']
            X[i,:] = X_distance[no_mask_id, :]
            X_random[i,:] = rX_distance[no_mask_id, :]

    y = np.asarray(select_trial_df['response'])

    # Split in development (train) and test / balanced for mask type
    indices = np.arange(len(select_trial_df))
    X_train, X_test, y_train, y_test, out_train_inds, out_test_inds = train_test_split(X, y, indices, test_size=0.1, stratify=select_trial_df['mask_type'].values, random_state=0)
    
    Xr_train = X_random[out_train_inds]
    Xr_test = X_random[out_train_inds]

    shell()

    # Simple model (without feature eliminatinon / grid search) + CV
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y=y_train)
    lr_basemodel = LogisticRegression(max_iter=5000, class_weight = {0:class_weights[0], 1:class_weights[1]})
    clf = make_pipeline(preprocessing.StandardScaler(), lr_basemodel)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scoring = ['accuracy', 'f1', 'recall', 'precision']

    # Evaluate lr model for different layers and for random network
    all_scores = []
    all_random_scores = []
    all_pred = []
    for i in range(n_layers):
        print(f"{layers[i]}")

        # Imagnet Network Activations - LR
        scores = cross_validate(clf, X_train[:, i].reshape(-1,1), y_train, scoring=scoring, cv=cv, return_train_score=True)
        all_scores.append(scores)
        mean_accuracy = np.mean(scores['test_accuracy'])
        mean_f1 = np.mean(scores['test_f1'])
        mean_recall = np.mean(scores['test_recall'])
        mean_precision = np.mean(scores['test_precision'])
        print(f"Mean accuracy: {mean_accuracy}, std {np.std(scores['test_accuracy'])}") #0.555
        print(f"Mean f1: {mean_f1}, std {np.std(scores['test_f1'])}") #0.557
        print(f"Mean recall: {mean_recall}, std {np.std(scores['test_recall'])} ") #0.476
        print(f"Mean precision: {mean_precision}, std {np.std(scores['test_precision'])}") #0.672

        # Cross validate prediction
        print(f"Cross val prediction")
        boot_res = []
        boot_inds = []
        for i in range(10):
            print(f"Boot {i}")
            boot_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
            boot_test_inds = []

            for train_index, test_index in boot_cv.split(X_train, y_train):
                boot_test_inds.append(test_index)

            boot_test_inds = np.array(boot_test_inds).reshape(-1,).tolist()
            boot_inds.append(boot_test_inds)

            y_train_pred = cross_val_predict(clf, all_X_train, y_train, cv=boot_cv)
            boot_res.append(y_train_pred)
        all_pred.append(boot_res)
        all_pred_ind.append(boot_inds)

        # Random network - LR
        random_scores = cross_validate(clf, Xr_train[:, i].reshape(-1,1), y_train, scoring=scoring, cv=cv, return_train_score=True)
        all_random_scores.append(random_scores)
        mean_accuracy = np.mean(scores['test_accuracy'])
        mean_f1 = np.mean(scores['test_f1'])
        mean_recall = np.mean(scores['test_recall'])
        mean_precision = np.mean(scores['test_precision'])
        print(f"Mean accuracy: {mean_accuracy}, std {np.std(scores['test_accuracy'])}") #0.555
        print(f"Mean f1: {mean_f1}, std {np.std(scores['test_f1'])}") #0.557
        print(f"Mean recall: {mean_recall}, std {np.std(scores['test_recall'])} ") #0.476
        print(f"Mean precision: {mean_precision}, std {np.std(scores['test_precision'])}") #0.672

    # Accuracy over layers plot
    test_accuracies = []
    test_f1s = []
    test_precisions = []
    test_recalls = []
    train_accuracies = []
    train_f1s = []
    train_precisions = []
    train_recalls = []
    test_r_accuracies = []
    test_r_f1s = []
    test_r_precisions = []
    test_r_recalls = []
    train_r_accuracies = []
    train_r_f1s = []
    train_r_precisions = []
    train_r_recalls = []
    for i in range(len(all_scores)):
        score = all_scores[i]
        r_score = all_random_scores[i]

        test_accuracies.append(np.mean(score['test_accuracy']))
        test_f1s.append(np.mean(score['test_f1']))
        test_precisions.append(np.mean(score['test_precision']))
        test_recalls.append(np.mean(score['test_recall']))

        test_r_accuracies.append(np.mean(r_score['test_accuracy']))
        test_r_f1s.append(np.mean(r_score['test_f1']))
        test_r_precisions.append(np.mean(r_score['test_precision']))
        test_r_recalls.append(np.mean(r_score['test_recall']))

        train_accuracies.append(np.mean(score['train_accuracy']))
        train_f1s.append(np.mean(score['train_f1']))
        train_precisions.append(np.mean(score['train_precision']))
        train_recalls.append(np.mean(score['train_recall']))

        train_r_accuracies.append(np.mean(r_score['train_accuracy']))
        train_r_f1s.append(np.mean(r_score['train_f1']))
        train_r_precisions.append(np.mean(r_score['train_precision']))
        train_r_recalls.append(np.mean(r_score['train_recall']))

    xlabels = layers 
    score_df = pd.DataFrame()
    score_df['test_accuracy'] = test_accuracies
    score_df['test_f1'] = test_f1s
    score_df['test_precision'] = test_precisions
    score_df['test_recall'] = test_recalls
    score_df['layer'] = xlabels
    score_df['test_r_accuracy'] = test_r_accuracies
    score_df['test_r_f1'] = test_r_f1s
    score_df['test_r_precision'] = test_r_precisions
    score_df['test_r_recall'] = test_r_recalls
    score_df['train_accuracy'] = train_accuracies
    score_df['train_f1'] = train_f1s
    score_df['train_precision'] = train_precisions
    score_df['train_recall'] = train_recalls
    score_df['train_r_accuracy'] = train_r_accuracies
    score_df['train_r_f1'] = train_r_f1s
    score_df['train_r_precision'] = train_r_precisions
    score_df['train_r_recall'] = train_r_recalls

    #  Plot performance over layers
    fig, ax =  plt.subplots(2,2, sharex=True, dpi=100, figsize=(14,7))
    fig.suptitle('Model performance')
    fig.supxlabel('Layer')
    fig.supylabel('Score')
    sns.lineplot(data=score_df, x='layer', y='train_accuracy', color='red', linestyle='--', ax=ax[0,0])
    sns.lineplot(data=score_df, x='layer', y='test_accuracy', color='red', ax=ax[0,0])
    sns.lineplot(data=score_df, x='layer', y='train_r_accuracy', color='firebrick', linestyle='--', ax=ax[0,0])
    sns.lineplot(data=score_df, x='layer', y='test_r_accuracy', color='firebrick', ax=ax[0,0])
    ax[0,0].set_title("Accuracy")
    ax[0,0].yaxis.label.set_visible(False)
    ax[0,0].xaxis.label.set_visible(False)
    ax[0,0].legend(handles=[
        Line2D([], [], marker='_', color="red", label="Pretrained"), 
        Line2D([], [], marker='_', color="firebrick", label="Random")], loc='upper right')
    sns.lineplot(data=score_df, x='layer', y='train_f1', color='blue', linestyle='--', ax=ax[0,1])
    sns.lineplot(data=score_df, x='layer', y='test_f1', color='blue', ax=ax[0,1])
    sns.lineplot(data=score_df, x='layer', y='train_r_f1', color='darkblue', linestyle='--', ax=ax[0,1])
    sns.lineplot(data=score_df, x='layer', y='test_r_f1', color='darkblue', ax=ax[0,1])
    ax[0,1].set_title("F1")
    ax[0,1].yaxis.label.set_visible(False)
    ax[0,1].xaxis.label.set_visible(False)
    ax[0,1].legend(handles=[
        Line2D([], [], marker='_', color="blue", label="Pretrained"),
        Line2D([], [], marker='_', color="darkblue", label="Random")], loc='upper right')
    sns.lineplot(data=score_df, x='layer', y='train_precision', color='palegreen', linestyle='--', ax=ax[1,0])
    sns.lineplot(data=score_df, x='layer', y='test_precision', color='palegreen', ax=ax[1,0])
    sns.lineplot(data=score_df, x='layer', y='train_r_precision', color='darkgreen', linestyle='--', ax=ax[1,0])
    sns.lineplot(data=score_df, x='layer', y='test_r_precision', color='darkgreen', ax=ax[1,0])
    ax[1,0].set_title("Precision")
    ax[1,0].yaxis.label.set_visible(False)
    ax[1,0].xaxis.label.set_visible(False)
    ax[1,0].set_xticklabels([1,2,3,4,5])
    ax[1,0].legend(handles=[
        Line2D([], [], marker='_', color="palegreen", label="Pretrained"),
        Line2D([], [], marker='_', color="darkgreen", label="Random")], loc='upper right')
    sns.lineplot(data=score_df, x='layer', y='train_recall', color='peachpuff', linestyle='--', ax=ax[1,1])
    sns.lineplot(data=score_df, x='layer', y='test_recall', color='peachpuff', ax=ax[1,1])
    sns.lineplot(data=score_df, x='layer', y='train_r_recall', color='darkorange', linestyle='--', ax=ax[1,1])
    sns.lineplot(data=score_df, x='layer', y='test_r_recall', color='darkorange', ax=ax[1,1])
    ax[1,1].set_title("Recall")
    ax[1,1].yaxis.label.set_visible(False)
    ax[1,1].xaxis.label.set_visible(False)
    ax[1,1].set_xticklabels(["1","2","3","4", "5", "all"])
    ax[1,1].legend(handles=[        
        Line2D([], [], marker='_', color="peachpuff", label="Pretrained"),
        Line2D([], [], marker='_', color="darkorange", label="Random")], loc='upper right')

    plt.figlegend(handles = [
        Line2D([], [], marker='_', color="black", label="Test score"),
        Line2D([], [], marker='_', color="black", label="Training score", linestyle="--")
        ])
    fig.tight_layout()
    file_name = os.path.join(wd, 'analysis/TrainTest_Score_across_layers_dis.png')
    fig.savefig(file_name)  

    # Inspect predictions --> bootstrap
    select_trial_df = select_trial_df.reset_index()

    # determine train df
    train_df = select_trial_df.iloc[out_train_inds]
    train_df = train_df.drop('level_0', 1)
    train_df = train_df.reset_index()

    for i in range(len(all_pred)):
        layer = xlabels[i]
        layer_preds = all_pred[i]
        boot_inds = all_pred_ind[i]

        for boot in range(10):
            layer_pred = layer_preds[boot]
            boot_ind = boot_inds[boot]
            order_pred = []
            for j in range(len(train_df)):
                idx = np.where(np.array(boot_ind) == j)
                order_pred.append(layer_pred[idx][0])

            select_trial_df[layer + f'_pred{boot}'] = order_pred

    # Calculate accuracy
    print("Calculating prediction accuracies")
    accuracies = {}
    n_masks = len(pd.unique(select_trial_df['mask_type']))
    for layer in xlabels:
        print(f"{layer}")
        boot_accuracies = np.zeros((n_masks, 10))
        for boot in range(5):
            print(f"Boot {boot}")
            correct = []
            for i in range(len(select_trial_df)):
                response = select_trial_df.iloc[i]['response']
                prediction = select_trial_df.iloc[i][layer + f'_pred{boot}']
                if response == prediction:
                    correct.append(1)
                else:
                    correct.append(0)
            select_trial_df[layer + f'_correct{boot}'] = correct
            accuracy_boot = select_trial_df.groupby('mask_type')[layer + f'_correct{boot}'].sum() / select_trial_df.groupby('mask_type')[layer + f'_correct{boot}'].count()
            boot_accuracies[:, boot] = accuracy_boot.reset_index()[layer + f'_correct{boot}']

        accuracy = np.mean(boot_accuracies, axis=1)
        # accuracy = select_trial_df.groupby('mask_type')[layer + '_correct'].sum() / select_trial_df.groupby('mask_type')[layer + '_correct'].count()
        accuracies[layer] = accuracy
    
    # Accuracy per mask plot
    fig, ax =  plt.subplots()
    colours = ['red', 'blue', 'orange', 'green', 'yellow', 'purple']
    for i in range(len(xlabels)):
        layer = xlabels[i]
        accuracy = accuracies[layer]
        accuracy_df = pd.DataFrame()
        accuracy_df['mask_type'] = pd.unique(select_trial_df['mask_type'])
        accuracy_df['score'] = accuracy
        sns.lineplot(data=accuracy_df.reset_index(), x='mask_type', y=f'score', color=colours[i])
    
    plt.figlegend(handles = [
        Line2D([], [], marker='_', color="red", label="Layer 1"),
        Line2D([], [], marker='_', color="blue", label="Layer 2"),
        Line2D([], [], marker='_', color="orange", label="Layer 3"),
        Line2D([], [], marker='_', color="green", label="Layer 4"),
        Line2D([], [], marker='_', color="yellow", label="Layer 5")])
    
    plt.ylabel("Accuracy")
    plt.xlabel("Layer")
    ax.set_xticklabels(['natural', 'scrambled', 'geometric', 'lines', 'blocked'])
    plt.title("Accuracy for different maks type")
    fig.tight_layout()
    file_name = os.path.join(wd, 'analysis/Accuracies-per-mask-dis.png')
    fig.savefig(file_name)  


    # Final evaluation also look at r2 squared + mean square error

# --------------- MAIN

# PCA fit
n_components = 500
# exp_features, pca_fits = fit_PCA(n_components)
# pca_fits = load_pca_fits()

# Sanity check - decode animacy
# animacy_features = feature_extraction_animacy(pca_fits, n_components)
# animacy_features = load_features(atype='animacy')
# animacy_perf = decode_animacy(animacy_features)
# inspect_predictions()

# Extract random features
# random_features, r_pca_fits = random_fit_PCA(n_components)

# Extract features all experimental images
exp_features = load_features(atype='exp')
random_features = load_features(atype='random')

# Preprocess behavioural data
bdata = preprocess_bdata()

# Set up log regression model
# logit_model(bdata, exp_features, random_features)

# Set up PLS model
PLS_model(bdata, exp_features, random_features)

# Distance analysis
# distance_analysis(bdata, exp_features, random_features)
