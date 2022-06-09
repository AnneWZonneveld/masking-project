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


def feature_extraction_random(pca_fits, n_components):
    """Extract features using PCA for only experimental target images"""
    print("Extracting features")

    return_nodes = {
        # node_name: user-specified key for output dict
        'relu': 'layer1',
        'layer1.2.relu': 'layer2', 
        'layer2.3.relu': 'layer3',
        'layer3.5.relu': 'layer4', 
        'layer4.2.relu': 'layer5'
    }

    # Randomly initialize model
    random_model = models.resnet50(pretrained=False)
    feature_extractor = create_feature_extractor(random_model, return_nodes=return_nodes)
    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']

    # Preprocess all experimental images to fit PCA
    n_images = len(all_images)
    img_paths = []
    for i in range(n_images):
        img_path = all_images[i]
        img_path = os.path.join(wd, img_path[53:])
        img_paths.append(img_path)
    
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

        output_dir = os.path.join(wd, f'analysis/random_features/{layer}')
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
            mask_type = mask_path.split('/')[-3]
            valid = data[data['index'] == trial_id]['valid_cue'].tolist()
            
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

    shell()

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

        # # Cross validate prediction
        # print(f"Cross val prediction")
        # y_train_pred = cross_val_predict(clf, X_train[:, :, i], y_train, cv=cv)
        # # y_train_pred_prob = cross_val_predict(clf, X_train, y_train, cv=cv, method = 'predict_proba')
        # cm = confusion_matrix(y_train, y_train_pred)
        # print("Confusion matrix:")
        # print(cm)
        # all_cms.append(cm)

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
        # # y_train_pred_prob = cross_val_predict(clf, X_train, y_train, cv=cv, method = 'predict_proba')
        # cm = confusion_matrix(y_train, ry_train_pred)
        # print("Confusion matrix random:")
        # print(cm)
        # all_random_cms.append(cm)
    
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

    score_df = pd.DataFrame()
    score_df['test_accuracy'] = test_accuracies
    score_df['test_f1'] = test_f1s
    score_df['test_precision'] = test_precisions
    score_df['test_recall'] = test_recalls
    score_df['layer'] = layers
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

    fig, ax = plt.subplots()
    sns.lineplot(data=score_df, x='layer', y='train_accuracy', color='red', linestyle='--')
    sns.lineplot(data=score_df, x='layer', y='test_accuracy', color='red')
    sns.lineplot(data=score_df, x='layer', y='train_r_accuracy', color='firebrick', linestyle='--')
    sns.lineplot(data=score_df, x='layer', y='test_r_accuracy', color='firebrick')
    sns.lineplot(data=score_df, x='layer', y='train_f1', color='blue', linestyle='--')
    sns.lineplot(data=score_df, x='layer', y='test_f1', color='blue')
    sns.lineplot(data=score_df, x='layer', y='train_r_f1', color='darkblue', linestyle='--')
    sns.lineplot(data=score_df, x='layer', y='test_r_f1', color='darkblue')
    sns.lineplot(data=score_df, x='layer', y='train_precision', color='palegreen', linestyle='--')
    sns.lineplot(data=score_df, x='layer', y='test_precision', color='palegreen')
    sns.lineplot(data=score_df, x='layer', y='train_r_precision', color='darkgreen', linestyle='--')
    sns.lineplot(data=score_df, x='layer', y='test_r_precision', color='darkgreen')
    sns.lineplot(data=score_df, x='layer', y='train_recall', color='peachpuff', linestyle='--')
    sns.lineplot(data=score_df, x='layer', y='test_recall', color='peachpuff')
    sns.lineplot(data=score_df, x='layer', y='train_r_recall', color='darkorange', linestyle='--')
    sns.lineplot(data=score_df, x='layer', y='test_r_recall', color='darkorange')
    plt.ylabel("Score")
    plt.title("Performance LR-model trained on pretrained verus random network activations across layers")

    from matplotlib.lines import Line2D 
    ax.legend(handles=[
        Line2D([], [], marker='_', color="red", label="Accuracy"), 
        Line2D([], [], marker='_', color="firebrick", label="Random accuracy"),
        Line2D([], [], marker='_', color="blue", label="F1"),
        Line2D([], [], marker='_', color="darkblue", label="Random F1"),
        Line2D([], [], marker='_', color="palegreen", label="Precision"),
        Line2D([], [], marker='_', color="darkgreen", label="Random precision"),
        Line2D([], [], marker='_', color="peachpuff", label="Recall"),
        Line2D([], [], marker='_', color="darkorange", label="Random recall"),
        Line2D([], [], marker='_', color="black", label="Test score"),
        Line2D([], [], marker='_', color="black", label="Training score", linestyle="--")
        ], loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.legend(['Accuracy', 'Random Accuracy', 'F1', 'Random F1','Precision', 'Random Precision', 'Recall', 'Random Recall'])
    plt.tight_layout()
    file_name = os.path.join(wd, 'analysis/TrainTest_Score_across_layers.png')
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


    # layers = ['layer5']
    # n_layers = len(layers)
    # n_features_og = X.shape[1]
    # feature_type = np.zeros((1, n_features_og))
    # for layer in layers:
    #     layer_select = np.linspace(0, n_features_og, num = n_layers + 1)
    #     # select = X[:, layer_select[0]:layer_select[1]]
    #     for activation in ['target', 'mask']:
    #         activation_select = np.linspace(layer_select[0], layer_select[1], num=3)

    # Assess fit 
    # clf = make_pipeline(preprocessing.StandardScaler(), fitted_model)
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # scoring = ['accuracy', 'f1', 'recall', 'precision']
    # scores = cross_validate(clf, X_train, y_train, scoring=scoring, cv=cv, return_estimator=True)

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

def MLP_model(data, exp_features, random_features):
        # -------------------------------- MLP
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import f1_score

    # Classifier pipeline
    MLP_model = MLPClassifier(activation='logistic', solver='sgd', random_state=0)
    clf = make_pipeline(preprocessing.StandardScaler(), MLP_model)

    # Hyper parameter tuning
    param_grid= {
        'mlpclassifier__hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
        'mlpclassifier__max_iter': [50,100,200],
        'mlpclassifier__alpha': [0.0001, 0.05],
        'mlpclassifier__learning_rate': ['invscaling','adaptive']}
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    scoring = ['accuracy', 'f1', 'precision', 'recall']
    gs = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1",
        n_jobs=-1,
        cv=cv
    )

    # Only on layer5 
    gs.fit(X_train[:, :, 4], y_train)
    results = gs.cv_results_
    best_fit = gs.best_estimator_
    best_score = gs.best_score_

    file_name = os.path.join(wd, "analysis/mlp_fit.pkl")
    pkl.dump(best_fit, open(file_name, "wb"))

    fig, ax = plt.subplots()
    plt.plot(best_fit.named_steps['mlpclassifier'].loss_curve_)
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.tight_layout()
    file_name = os.path.join(wd, f'analysis/MLP_bestfit_loss.png')
    fig.savefig(file_name)  

    # Manual cv 
    # Test
    sc = preprocessing.StandardScaler()
    scaler = sc.fit(X_train[:,:, i])
    X_train_s = scaler.transform(X_train[:, :, i])
    X_test_s = scaler.transform(X_test[:, :, i])
    MLP_model.fit(X_train_s, y_train)
    y_pred = MLP_model.predict(X_test_s)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

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
    X1_single = np.zeros((n_mask_trials, n_components, n_layers))
    X2_single = np.zeros((n_mask_trials, n_components, n_layers))

    rX1_single = np.zeros((n_mask_trials, n_components, n_layers))
    rX2_single = np.zeros((n_mask_trials, n_components, n_layers))

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

            X1_single[no_mask_id, :, :] = target_activations
            rX1_single[no_mask_id, :, :] = rtarget_activations

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

            X2_single[no_mask_id, :, :] = mask_activations
            rX2_single[no_mask_id, :, :] = rmask_activations

            # get response for all participants (average?)
            responses = data[data['index'] == trial_id]['correct'].tolist() #anwer or correct?
            subject_nrs = data[data['index'] == trial_id]['subject_nr'].tolist()
            mask_type = mask_path.split('/')[-3]

            tmp['index'] = [trial_id for i in range(len(responses))]
            tmp['response'] = responses
            tmp['subject_nr'] = subject_nrs
            tmp['target_path'] = [target_path for i in range(len(responses))]
            tmp['mask_path'] = [mask_path for i in range(len(responses))]
            tmp['mask_type'] = [mask_type for i in range(len(responses))]
            tmp['mask_activation'] = [mask_activations for i in range(len(responses))]
            tmp['target_activation'] = [target_activations for i in range(len(responses))]
            tmp['rmask_activation'] = [rmask_activations for i in range(len(responses))]
            tmp['rtarget_activation'] = [rtarget_activations for i in range(len(responses))]

            trial_df = pd.concat([trial_df, tmp], ignore_index=True)

            no_mask_id += 1
    
    X1_single_scaled= np.zeros((n_mask_trials, n_components, n_layers))
    X2_single_scaled = np.zeros((n_mask_trials, n_components, n_layers))
    rX1_single_scaled = np.zeros((n_mask_trials, n_components, n_layers))
    rX2_single_scaled = np.zeros((n_mask_trials, n_components, n_layers))

    X_distance = np.zeros((n_mask_trials, n_layers))
    rX_distance = np.zeros((n_mask_trials, n_layers))

    for i in range(n_layers):
        scaler_1 = preprocessing.StandardScaler().fit(X1_single[:,:, i])
        X1_single_scaled[:,:, i] = scaler_1.transform(X1_single[:,:, i])
        scaler_2 = preprocessing.StandardScaler().fit(X2_single[:,:, i])
        X2_single_scaled[:,:, i] = scaler_2.transform(X2_single[:,:, i])

        r_scaler_1 = preprocessing.StandardScaler().fit(rX1_single[:,:, i])
        rX1_single_scaled[:,:, i] = r_scaler_1.transform(rX1_single[:,:, i])
        r_scaler_2 = preprocessing.StandardScaler().fit(rX2_single[:,:, i])
        rX2_single_scaled[:,:, i] = r_scaler_2.transform(rX2_single[:,:, i])

    for i in range(n_mask_trials):
        for j in range(n_layers):
            target = X1_single_scaled[i, :, j]
            mask = X2_single_scaled[i, :, j]
            r_target = rX1_single_scaled[i, :, j]
            r_mask = rX2_single_scaled[i, :, j]

            distance = spatial.distance.correlation(target, mask)
            r_distance = spatial.distance.correlation(r_target, r_mask)

            X_distance[i, j] = distance
            rX_distance[i, j] = r_distance

    distance_df = pd.DataFrame()
    for i in range(n_mask_trials):
        tmp = pd.DataFrame()
        distance = X_distance[i, :]
        r_distance = rX_distance[i, :]
        distances = [distance for i in range(len(responses))]
        r_distances = [r_distance for i in range(len(responses))]
        tmp['distance'] = distances
        tmp['r_distance'] = r_distances
        distance_df = pd.concat([distance_df, tmp], ignore_index=True)
    
    shell()
    trial_df['distance'] = distance_df['distance']
    trial_df['r_distance'] = distance_df['r_distance']

    cors = []
    r_cors = []
    for j in range(n_layers):
        distances = []
        r_distances = []
        for i in range(len(trial_df)):
            distances.append(trial_df['distance'].iloc[i][j])
            r_distances.append(trial_df['r_distance'].iloc[i][j])
        cor = stats.pearsonr(trial_df['response'], distances)
        r_cor = stats.pearsonr(trial_df['response'], r_distances)
        cors.append(cor)
        r_cors.append(r_cor)

    # Final evaluation also look at r2 squared + mean square error

# --------------- MAIN

# PCA fit
n_components = 500
# exp_features, pca_fits = fit_PCA(n_components)
pca_fits = load_pca_fits()

# Sanity check - decode animacy
# animacy_features = feature_extraction_animacy(pca_fits, n_components)
# animacy_features = load_features(atype='animacy')
# animacy_perf = decode_animacy(animacy_features)
# inspect_predictions()

# Extract random features
# random_features = feature_extraction_random(pca_fits, n_components)

# Extract features all experimental images
exp_features = load_features(atype='exp')
random_features = load_features(atype='random')

# Preprocess behavioural data
bdata = preprocess_bdata()

# Set up log regression model
# logit_model(bdata, exp_features, random_features)

# Distance analysis
distance_analysis(bdata, exp_features, random_features)
