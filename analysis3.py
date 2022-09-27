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
import time 
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
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, StratifiedKFold, GridSearchCV, cross_validate, train_test_split, cross_val_predict, GroupShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from yellowbrick.model_selection import RFECV
from matplotlib.lines import Line2D     
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix, r2_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import make_pipeline, Pipeline
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import spearmanr, ttest_1samp, pearsonr, ttest_rel, wilcoxon
import scikit_posthocs as sp


wd = '/home/c11645571/masking-project'
trial_file = pd.read_csv(os.path.join(wd, 'help_files', 'selection_THINGS.csv'))  
concept_selection = pd.read_csv(os.path.join(wd, "help_files", "concept_selection.csv"), sep=';', header=0) 
all_targets = pd.unique(trial_file['ImageID']).tolist()
all_masks = pd.unique(trial_file['mask_path']).tolist()
all_masks = [path for path in all_masks if path != 'no_mask']
all_images = pd.unique(all_targets + all_masks).tolist()

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

transform  = T.Compose([
            T.ToPILImage(),
            T.CenterCrop(512),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=0., std=1.) # correct parameters?
        ])


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

sns.set_context('paper', rc={'font.size': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                             'figure.titleweight': 'bold', 'axes.labelsize': 10, 'axes.titlesize': 10})


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


def extract(model, m_type):
    """Extract features from model"""
    
    return_nodes = {
        # node_name: user-specified key for output dict
        'relu': 'layer1',
        'layer1.2.relu': 'layer2', 
        'layer2.3.relu': 'layer3',
        'layer3.5.relu': 'layer4', 
        'layer4.2.relu': 'layer5'
    }

    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    # Preprocess all experimental images to 
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
    
    # Create text file with all image paths
    df = pd.DataFrame (new_paths, columns = ['path'])
    df.to_csv(os.path.join(wd, 'analysis', 'image_paths_exp.csv')) 

    # Extract features 
    print(f"Extracting features")
    imgs = torch.from_numpy(imgs).type(torch.DoubleTensor)
    feature_dict = feature_extractor(imgs) 
    
    del imgs

    # Tranform; average over spatial dimension
    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']
    t_features  = {}

    for layer in layers:
        features = feature_dict[layer].detach().numpy()
        av_features = np.mean(features, axis=(2,3))
        t_features[layer] = av_features

    # Save features
    if m_type == 'pretrained':
        folder = 'exp_features'
    elif m_type == 'random':
        folder = 'random_features'

    for layer in layers:

        output_dir = os.path.join(wd, f'analysis/{folder}/{layer}')
        layer_features = t_features[layer]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
            with open(os.path.join(output_dir, 'features.npy'), 'wb') as f:
                np.save(f, layer_features)

    return t_features


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


def PLS_model(data, features, exp = True, thres = 0):
    file_dir = os.path.join(wd, 'analysis', 'image_paths_exp.csv') # should be cc1
    image_paths = pd.read_csv(file_dir)['path'].tolist()
    concepts = pd.unique(concept_selection['concept']).tolist()

    n_trials = len(trial_file)
    trial_df = pd.DataFrame()
    no_mask_trial_df = pd.DataFrame()
   
    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']
    n_layers = len(layers)
    no_mask = [i for i in range(len(trial_file)) if trial_file.iloc[i]['mask_type']=='no_mask']
    n_mask_trials = n_trials - len(no_mask)

    X = {
        'layer1': np.zeros((n_mask_trials, 64*2)),
        'layer2': np.zeros((n_mask_trials, 64*2)),
        'layer3': np.zeros((n_mask_trials, 128*2)),
        'layer4': np.zeros((n_mask_trials, 256*2)),
        'layer5': np.zeros((n_mask_trials, 512*2)),
        }

    X1 = {
        'layer1': np.zeros((n_mask_trials, 64)),
        'layer2': np.zeros((n_mask_trials, 64)),
        'layer3': np.zeros((n_mask_trials, 128)),
        'layer4': np.zeros((n_mask_trials, 256)),
        'layer5': np.zeros((n_mask_trials, 512)),
        }

    X2 = {
        'layer1': np.zeros((n_mask_trials, 64)),
        'layer2': np.zeros((n_mask_trials, 64)),
        'layer3': np.zeros((n_mask_trials, 128)),
        'layer4': np.zeros((n_mask_trials, 256)),
        'layer5': np.zeros((n_mask_trials, 512)),
        }


    print("Creating trial df")
    mask_id = 0
    mask_trials = []

    for j in range(len(trial_file)):

        tmp = pd.DataFrame()
        
        trial = trial_file.iloc[j]
        trial_id  = j

        # Target path
        target_path = trial['ImageID']
        target_path = os.path.join(wd, target_path[53:])

        # Check if trial is mask trial
        mask_path = trial['mask_path']

        if mask_path == 'no_mask':
            mask_type = 'no mask'
        else:
            mask_trials.append(j)
            
            # get according activations
            target_index = image_paths.index(target_path)
            mask_path = os.path.join(wd, mask_path[53:])
            mask_index = image_paths.index(mask_path)
            for i in range(len(layers)):
                layer = layers[i]
                target_activation = features[layer][target_index, :]
                mask_activation = features[layer][mask_index, :]
                X1[layer][mask_id, :] = target_activation
                X2[layer][mask_id, :] = mask_activation
                X[layer][mask_id, :] =  np.concatenate((target_activation, mask_activation))

            mask_type = mask_path.split('/')[-3]
            if mask_type == 'masks':
                mask_type = mask_path.split('/')[-2]

            mask_id += 1
   
        # get response for all participants
        responses = data[data['index'] == trial_id]['answer'].tolist() #anwer or correct?
        correct = data[data['index'] == trial_id]['correct'].tolist() 
        subject_nrs = data[data['index'] == trial_id]['subject_nr'].tolist()
        valid = data[data['index'] == trial_id]['valid_cue'].tolist()
        
        tmp['index'] = [trial_id for i in range(len(responses))]
        tmp['response'] = responses
        tmp['valid'] = valid
        tmp['correct'] = correct
        tmp['subject_nr'] = subject_nrs
        tmp['target_path'] = [target_path for i in range(len(responses))]
    
        if mask_type != 'no_mask':
            tmp['mask_path'] = [mask_path for i in range(len(responses))]
            tmp['mask_type'] = [mask_type for i in range(len(responses))]
        else:
            tmp['mask_path'] = tmp['mask_type'] = [float("NaN") for i in range(len(responses))]

        trial_df = pd.concat([trial_df, tmp], ignore_index=True)
    
    all_X = np.concatenate((X['layer1'], X['layer2'], X['layer3'], X['layer4'], X['layer5']), axis=1)
    X['all'] = all_X

    layers_oi = layers + ['all']

    if exp == True:
        file_path = os.path.join(wd, 'analysis/stats/X')
        file = open(file_path, 'wb')
        pkl.dump(X, file)
        file.close()
        file_path = os.path.join(wd, 'analysis/stats/X1')
        file = open(file_path, 'wb')
        pkl.dump(X1, file)
        file.close()
        file_path = os.path.join(wd, 'analysis/stats/X2')
        file = open(file_path, 'wb')
        pkl.dump(X2, file)
        file.close()
    else:
        file_path = os.path.join(wd, 'analysis/stats/X_random')
        file = open(file_path, 'wb')
        pkl.dump(X, file)
        file.close()
        file_path = os.path.join(wd, 'analysis/stats/X1_random')
        file = open(file_path, 'wb')
        pkl.dump(X1, file)
        file.close()
        file_path = os.path.join(wd, 'analysis/stats/X2_random')
        file = open(file_path, 'wb')
        pkl.dump(X2, file)
        file.close()
    

    # Mask / no mask mean accuracy 
    type_means = trial_df.groupby(['mask_type'])['correct'].mean()
    
    trial_means = trial_df.groupby(['index'])['correct'].mean()
    mask_types = trial_file['mask_type']
    GA_df = pd.concat([trial_means, mask_types], ignore_index=True, axis=1)
    GA_df = GA_df.rename(columns={0: "Accuracy", 1: "Mask type"})

    # Plot accuracies
    fig, axes = plt.subplots(1, figsize=(9, 5))
    masks_ordered = ['no_mask', '1_natural', '6_blocked', '2_scrambled', '5_lines', '4_geometric']
    GA_df["Mask type"] = pd.Categorical(GA_df["Mask type"], masks_ordered)
    sns.set_palette('colorblind')
    sns.set_style('ticks')
    # g = sns.stripplot(data=GA_df, x="Mask type", y="Accuracy", clip_on=False, alpha=0.5, jitter=0.3)
    g = sns.swarmplot(data=GA_df, x="Mask type", y="Accuracy", clip_on=False, alpha=0.5)
    g.set_ylim([-0.1, 1])
    sns.despine(offset=10)
    plt.tight_layout()
    
    g = sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="Mask type",
            y="Accuracy",
            data=GA_df,
            showfliers=False,
            showbox=False,
            showcaps=False)

    g.set_xticklabels(['no mask', 'natural', 'blocked', 'scrambled', 'lines', 'geometric'])
    g.set_ylim(bottom=0, top=1)
    image_paths = os.path.join(wd, 'analysis/Accuracy-per-mask.png')
    plt.savefig(image_paths)
    plt.clf()

    #  -------------------- ANOVA
    # Test normality --> all not normally distributed
    shapiro_res = []
    for i in range(len(masks_ordered)):
        mask_type = masks_ordered[i]
        print(f"{mask_type}")
        tmp = GA_df[GA_df['Mask type']==mask_type]
        shapiro = stats.shapiro(tmp['Accuracy'])
        shapiro_res.append(shapiro)
        print(shapiro)
    df = len(tmp) - 1
    print(f"DF: {df}")

    # Test homoscedasticity --> violated
    stat, p_value = stats.levene( 
        GA_df[GA_df['Mask type']=='1_natural']['Accuracy'],
        GA_df[GA_df['Mask type']=='no_mask']['Accuracy'],
        GA_df[GA_df['Mask type']=='6_blocked']['Accuracy'],
        GA_df[GA_df['Mask type']=='2_scrambled']['Accuracy'], 
        GA_df[GA_df['Mask type']=='5_lines']['Accuracy'], 
        GA_df[GA_df['Mask type']=='4_geometric']['Accuracy'], 
        center='mean')
    print("Levene's")
    print(f"stat {stat}, p {p_value}")

    # Kruskal Wallice test --> significant
    stat, p_value  = stats.kruskal( 
        GA_df[GA_df['Mask type']=='1_natural']['Accuracy'],
        GA_df[GA_df['Mask type']=='no_mask']['Accuracy'],
        GA_df[GA_df['Mask type']=='6_blocked']['Accuracy'],
        GA_df[GA_df['Mask type']=='2_scrambled']['Accuracy'], 
        GA_df[GA_df['Mask type']=='5_lines']['Accuracy'], 
        GA_df[GA_df['Mask type']=='4_geometric']['Accuracy'])
    print("Kruskal")
    print(f"stat {stat}, p {p_value}")

    # Post hoc Dunn test --> do in R for Z-stat?
    print("Post hoc Dunn")
    p_values = sp.posthoc_dunn(GA_df, 'Accuracy', 'Mask type','bonferroni')
    print(p_values)
    p_values < 0.05


    # Calc mask efficacy
    efficacies = []
    for i in range(n_mask_trials):
        mask_id = mask_trials[i]
        target = trial_file.iloc[mask_id]['ImageID']
        no_mask_mean_id = np.array(trial_file[(trial_file['ImageID'] == target) & (trial_file['mask_type'] == 'no_mask')].reset_index()['index'])[0]
        no_mask_mean = GA_df.iloc[no_mask_mean_id]['Accuracy']

        mask_ef = no_mask_mean - GA_df.iloc[mask_id]['Accuracy']
        efficacies.append(mask_ef)

    GA_df_masks = GA_df.iloc[mask_trials]
    GA_df_masks['Efficacy'] = efficacies
    GA_df_masks = GA_df_masks.reset_index()

    file_path = os.path.join(wd, 'analysis/stats/')
    if not os.path.exists(file_path):
            os.makedirs(file_path)
    GA_df_masks.to_csv(os.path.join(file_path, 'GA_df_masks.csv'), index=False)

    # Plot mask efficacies
    masks_ordered = ['1_natural', '6_blocked', '2_scrambled', '5_lines', '4_geometric']
    GA_df_masks["Mask type"] = pd.Categorical(GA_df_masks["Mask type"], masks_ordered)
    g = sns.swarmplot(data=GA_df_masks, x="Mask type", y="Efficacy", clip_on=False, alpha=0.5)
    sns.set_palette('colorblind')
    sns.set_style('ticks')
    sns.despine(offset=10)
    plt.tight_layout()
    
    g = sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="Mask type",
            y="Efficacy",
            data=GA_df_masks,
            showfliers=False,
            showbox=False,
            showcaps=False)

    g.set_xticklabels(['natural', 'blocked', 'scrambled', 'lines', 'geometric'])
    g.set_ylim([-0.1, 1])
    plt.axhline(y=0.0, color='gray', linestyle='--', clip_on = False)
    image_path = os.path.join(wd, 'analysis/Efficacy-per-mask.png')
    plt.savefig(image_path)
    plt.clf()

    shell()

    # Set up pipeline
    from sklearn.cross_decomposition import PLSRegression
    lr = PLSRegression(n_components=20)
    clf = make_pipeline(preprocessing.StandardScaler(), lr)
    
    y = np.asarray(GA_df_masks['Efficacy'])
    preds = np.zeros((n_mask_trials, len(layers_oi)))
    for i in range(len(layers_oi)):
        layer = layers_oi[i]
        print(f"CV for {layer}")
        
        pred = cross_val_predict(clf, X[layer], y, cv=n_mask_trials)
        preds[:, i] = np.squeeze(pred)   

    file_path = os.path.join(wd, 'analysis/predictions/')
    if not os.path.exists(file_path):
            os.makedirs(file_path)

    if exp == True:
        np.save(os.path.join(file_path, f"PLS_predictions.npy"), preds)
    else: 
        np.save(os.path.join(file_path, f"PLS_predictions_random.npy"), preds)

    # Inspect model
    mins = []
    maxs = []
    for i in range(n_layers):
        layer = layers[i]
        min = np.min(preds[:, i])
        max = np.max(preds[:, i])
        mins.append(min)
        maxs.append(max)
    print(f"max: {maxs}")
    print(f"min: {mins}")

    # MAE error - per layer
    # file_path = os.path.join(wd, 'analysis/predictions/')
    # preds = glob.glob(os.path.join(file_path, 'PLS_predictions.npy'))[0]
    # preds = np.load(preds)

    MAEs = []
    r2s = []
    for i in range(n_layers + 1):
        print(f'Layer {i}')

        y_pred = preds[:, i]
        error = MAE(y, y_pred)
        r2 = (pearsonr(y, y_pred)[0])**2
        MAEs.append(error)
        r2s.append(r2)
    
    MAE_df = pd.concat([pd.DataFrame(MAEs), pd.DataFrame(layers_oi), pd.DataFrame(r2s)], axis=1)
    MAE_df.columns = ['MAE', 'layer', 'r2']

    # Save 
    file_path = os.path.join(wd, 'analysis/stats/')
    if not os.path.exists(file_path):
            os.makedirs(file_path)
    MAE_df.to_csv(os.path.join(file_path, 'MAE_per_layer.csv'), index=False)

    # MAE plot
    fig, axes = plt.subplots(1, figsize=(4,3))
    sns.set_palette('colorblind')
    sns.set_style("ticks")
    sns.pointplot(data=MAE_df, x="layer", y="MAE")
    sns.despine(offset=10)
    xlabels = ['layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5', 'all']
    plt.xticks(np.arange(len(xlabels)),xlabels)
    plt.tight_layout()

    if exp == True:
        image_path = os.path.join(wd, f'analysis/MAEs-layers.png')
    else:
        image_path = os.path.join(wd, f'analysis/MAEs-layers-random.png')
    plt.savefig(image_path)
    plt.clf()

    # R2 plot
    sns.set_palette('colorblind')
    sns.set_style("ticks")
    sns.pointplot(data=MAE_df, x="layer", y="r2")
    sns.despine(offset=15)
    xlabels = ['layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5', 'all']
    plt.xticks(np.arange(len(xlabels)),xlabels)
    # plt.ylim([0.1, 0.35])
    plt.tight_layout()
    if exp == True:
        image_path = os.path.join(wd, f'analysis/r2-layers.png')
    else:
        image_path = os.path.join(wd, f'analysis/r2-layers_random.png')

    plt.savefig(image_path)
    plt.clf()

    # Evaluate MAE per mask type
    MAE_mask_df = pd.DataFrame()
    mask_MAEs = []
    mask_r2s = []
    layer_nr = layers_oi * len(masks_ordered)
    mask_nr = []

    for j in range(len(masks_ordered)):

        mask_type = masks_ordered[j]
        mask_name = mask_type.split("_")[1]
        mask_ids = np.array(GA_df_masks[GA_df_masks['Mask type']== mask_type].index)

        for i in range(n_layers + 1):

            layer = layers_oi[i]

            mask_pred = preds[mask_ids, i]
            y_mask = y[mask_ids]

            r2 = (pearsonr(y_mask, mask_pred)[0])**2
            error = MAE(y_mask, mask_pred)
            mask_MAEs.append(error)
            mask_r2s.append(r2)

            mask_nr.append(mask_name)
    
    MAE_mask_df['MAE'] = mask_MAEs
    MAE_mask_df['r2'] = mask_r2s
    MAE_mask_df['Mask'] = mask_nr
    MAE_mask_df['Layer'] = layer_nr

    # Save 
    file_path = os.path.join(wd, 'analysis/stats/')
    if not os.path.exists(file_path):
            os.makedirs(file_path)
    MAE_mask_df.to_csv(os.path.join(file_path, 'MAE_per_mask.csv'), index=False)

    # Plot MAE per mask per layer
    fig, axes = plt.subplots(1, figsize=(5,4))
    sns.pointplot(data=MAE_mask_df, x='Layer', y='MAE', hue='Mask')
    sns.despine(offset=10)
    # plt.ylim([0.07, 0.14])
    xlabels = ['layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5', 'all']
    plt.xticks(np.arange(len(xlabels)),xlabels)
    plt.xlabel('')
    plt.tight_layout()
    image_path = os.path.join(wd, 'analysis/MAEs-per-mask.png')
    plt.savefig(image_path)
    plt.clf()

    # Plot r2 per mask per layer
    sns.pointplot(data=MAE_mask_df, x='Layer', y='r2', hue='Mask')
    sns.despine(offset=10)
    plt.ylim([0.0, 0.5])
    xlabels = ['layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5', 'all']
    plt.xticks(np.arange(len(xlabels)),xlabels)
    plt.xlabel('')
    plt.tight_layout()
    image_path = os.path.join(wd, 'analysis/r2-per-mask.png')
    plt.savefig(image_path)
    plt.clf()


    # Prediction per image for best layer (= layer5)
    tmp_df_1 = pd.DataFrame(preds[:, 4]).reset_index()
    tmp_df_2 = pd.DataFrame(y).reset_index()

    sns.scatterplot(x='index', y=0, data=tmp_df_2, markers='d', color ='b', alpha=0.6)
    sns.scatterplot(x='index', y=0, data=tmp_df_1, markers='d', color ='r')

    sns.despine(offset=10)
    plt.xticks([])
    plt.xlabel('Unique trial')
    plt.ylabel('Mask efficacy')

    handles = [Line2D([0], [0], color='blue', lw=3, linestyle='-'), Line2D([0], [0], color='r', lw=3, linestyle='-'),]
    labels =[ 'observed mask efficacy', 'layer 4 prediction']

    l = plt.legend(handles, labels, loc=1,
                borderaxespad=0., frameon=False)

    plt.tight_layout()
    image_path = os.path.join(wd, 'analysis/prediction-per-image.png')
    plt.savefig(image_path)
    plt.clf()


    # Cor plot
    fig, axes = plt.subplots(1, 6, sharex=True, sharey=True, dpi=100, figsize=(14,7))
    for i in range(n_layers + 1):
        layer = layers_oi[i]

        pred_df = pd.concat([pd.DataFrame(preds[:, i]), pd.DataFrame(y)], axis=1)
        pred_df.columns = ['y_pred', 'y']

        sns.regplot(x='y', y='y_pred', data=pred_df, color='red', ci=None, ax=axes[i], scatter_kws={'alpha':0.3})
        axes[i].set_title(f"{layer}")

        sns.set_style({"xtick.direction": "in","ytick.direction": "in", "font_scale": 15})
        sns.despine()

        # Correlate 
        n_ivs = X[layer].shape[1]
        r, p = pearsonr(pred_df['y'], pred_df['y_pred'])
        r2 = r**2
        # adj_r2 = 1 - (1-r2) *((n_mask_trials -1)/(n_mask_trials - n_ivs))
        print(f"{layer}")
        print(f'r = {r:.3f}\np = {p}\nr2 = {r**2:.3f}')
        # print(f'adj_r2 = {adj_r2:.3f}')

        axes[i].set_xlabel(f'r^2 = {r**2:.3f}')
        axes[i].set_ylabel('')
        axes[i].set_xticks([],size=35)
        axes[i].set_yticks([], size=35)

    fig.supxlabel('Mask efficacy')
    fig.supylabel('Predicted mask efficacy')
    image_path = os.path.join(wd, 'analysis/predictions/corplot.png')
    plt.savefig(image_path)

    

def model_per_mask():
    shell()
    
    from sklearn.cross_decomposition import PLSRegression

    # Load GA_df_masks
    file_path = os.path.join(wd,  'analysis/stats/GA_df_masks.csv')
    GA_df_masks = pd.read_csv(file_path)
    y = np.asarray(GA_df_masks['Efficacy'])

    # Load X 
    file_path = os.path.join(wd,  'analysis/stats/')
    X = glob.glob(os.path.join(file_path, 'X'))[0]
    file = open(X, 'rb')
    X = pkl.load(file)
    file.close()

    # Set up pipeline
    lr = PLSRegression(n_components=20)
    clf = make_pipeline(preprocessing.StandardScaler(), lr)
  
    n_mask_trials = len(GA_df_masks)    
    masks_ordered = ['1_natural', '5_lines', '6_blocked', '4_geometric', '2_scrambled']
    layers_oi = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'all']

    preds = np.zeros((int(n_mask_trials/len(masks_ordered)), len(masks_ordered), len(layers_oi)))
    MAE_masks = np.zeros((len(masks_ordered), len(layers_oi)))
    r2_masks = np.zeros((len(masks_ordered), len(layers_oi)))

    for i in range(len(layers_oi)):
        layer = layers_oi[i]
        print(f" {layer}")

        for j in range(len(masks_ordered)):
            mask_type = masks_ordered[j]
            print(f" {mask_type}")

            mask_ids = np.array(GA_df_masks[GA_df_masks['Mask type']== mask_type].index)
            X_select = X[layer][mask_ids]
            y_select= y[mask_ids]

            mask_pred = np.squeeze(cross_val_predict(clf, X_select, y_select, cv=X_select.shape[0]))
            preds[:, j, i] = mask_pred

            error = MAE(y_select, mask_pred)
            MAE_masks[j, i] = error
            r2 = pearsonr(y_select, mask_pred)[0]
            MAE_masks[j, i] = error
            r2_masks[j, i] = r2
    
    file_path = os.path.join(wd, 'analysis/predictions/')
    if not os.path.exists(file_path):
            os.makedirs(file_path)

    np.save(os.path.join(file_path, "PLS_per_mask.npy"), preds)

    # Evaluate MAE per mask type
    preds = glob.glob(os.path.join(file_path,  "MAE_per_mask_fitted.npy"))[0]
    preds = np.load(preds)
    MAE_mask_df = pd.DataFrame()
    mask_MAEs = []
    mask_r2s = []
    layer_nr = layers_oi * len(masks_ordered)
    mask_nr = []

    for j in range(len(masks_ordered)):

        mask_type = masks_ordered[j]
        mask_name = mask_type.split("_")[1]

        for i in range(len(layers_oi)):

            layer = layers_oi[i]
            mask_MAE = MAE_masks[j, i]
            mask_r2 = r2_masks[j, i]
            mask_MAEs.append(mask_MAE)
            mask_r2s.append(mask_r2)
            mask_nr.append(mask_name)
    
    MAE_mask_df['MAE'] = mask_MAEs
    MAE_mask_df['r2'] = mask_r2s
    MAE_mask_df['Mask'] = mask_nr
    MAE_mask_df['Layer'] = layer_nr

    # Save 
    file_path = os.path.join(wd, 'analysis/stats/')
    if not os.path.exists(file_path):
            os.makedirs(file_path)
    MAE_mask_df.to_csv(os.path.join(file_path, 'MAE_fitted_per_mask.csv'), index=False)

    # Plot MAE per mask per layer
    fig, axes = plt.subplots(1, figsize=(5,4))
    sns.set_palette('tab10')
    sns.set_style("white")
    sns.pointplot(data=MAE_mask_df, x='Layer', y='MAE', hue='Mask')
    sns.despine(offset=10)
    xlabels = ['layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5', 'all']
    plt.xticks(np.arange(len(xlabels)),xlabels)
    plt.xlabel('')
    plt.tight_layout()
    image_path = os.path.join(wd, 'analysis/MAEs-fitted-per-mask.png')
    plt.savefig(image_path)
    plt.clf()

    # Plot R2 per mask per layer
    sns.set_palette('tab10')
    sns.set_style("white")
    sns.pointplot(data=MAE_mask_df, x='Layer', y='r2', hue='Mask')
    sns.despine(offset=10)
    xlabels = ['layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5', 'all']
    plt.xticks(np.arange(len(xlabels)),xlabels)
    plt.xlabel('')
    # plt.ylim([-0.2, 0.5])
    plt.tight_layout()
    image_path = os.path.join(wd, 'analysis/r2-fitted-per-mask.png')
    plt.savefig(image_path)
    plt.clf()


def calc_distance(features):
    shell()

    # Load X 
    file_path = os.path.join(wd,  'analysis/stats/')
    X1 = glob.glob(os.path.join(file_path, 'X1'))[0]
    file = open(X1, 'rb')
    X1= pkl.load(file)
    file.close()
    X2 = glob.glob(os.path.join(file_path, 'X2'))[0]
    file = open(X2, 'rb')
    X2= pkl.load(file)
    file.close()

    # Load GA_df_masks
    file_path = os.path.join(wd,  'analysis/stats/GA_df_masks.csv')
    GA_df_masks = pd.read_csv(file_path)
    y = np.asarray(GA_df_masks['Efficacy'])

    # Load predictions
    file_path = os.path.join(wd, 'analysis/predictions')
    preds = glob.glob(os.path.join(file_path, 'lr_predictions.npy'))[0]
    preds = np.load(preds)

    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']
    n_mask_trials = X1['layer1'].shape[0]

    cors = {
        'layer1': np.zeros((n_mask_trials)),
        'layer2': np.zeros((n_mask_trials)),
        'layer3': np.zeros((n_mask_trials)),
        'layer4': np.zeros((n_mask_trials)),
        'layer5': np.zeros((n_mask_trials)),
        }
    
    resids =  {
        'layer1': np.zeros((n_mask_trials)),
        'layer2': np.zeros((n_mask_trials)),
        'layer3': np.zeros((n_mask_trials)),
        'layer4': np.zeros((n_mask_trials)),
        'layer5': np.zeros((n_mask_trials)),
        }

    for i in range(len(layers)):
        layer = layers[i]

        for trial in range(n_mask_trials):
            cor = spearmanr(X1[layer][trial, :], X2[layer][trial, :])[0]
            cors[layer][trial] = cor
            resid = abs(y[trial] - preds[trial, i])
            resids[layer][trial] = resid
    
    file_path = os.path.join(wd, 'analysis/stats/cors')
    file = open(file_path, 'wb')
    pkl.dump(cors, file)
    file.close()

    # Performance accuracy / similarity plot
    fig, axes = plt.subplots(1, 5,dpi=100, figsize=(14,7))
    for i in range(len(layers)):
        layer = layers[i]

        reg_df = pd.concat([pd.DataFrame(cors[layer]), pd.DataFrame(y)], axis=1)
        reg_df.columns = ['Similarity', 'Residuals']

        sns.regplot(x='Similarity', y='Residuals', data=reg_df, color='blue', ci=None, ax=axes[i], scatter_kws={'alpha':0.3})
        axes[i].set_title(f"{layer}")

        sns.set_style({"xtick.direction": "in","ytick.direction": "in", "font_scale": 15})
        sns.despine()

        # Correlate 
        r, p = pearsonr(reg_df['Residuals'], reg_df['Similarity'])
        print(f"{layer}")
        print(f'r = {r:.3f}\np = {p}\nr2 = {r**2:.3f}')

        axes[i].set_xlabel(f'r = {r:.3f}, p = {p:.3f}')
        axes[i].set_ylabel('')
        # axes[i].set_xticks([],size=35)
        # axes[i].set_yticks([], size=35)

    fig.supxlabel('Feature similarity')
    fig.supylabel('Efficacy')
    image_path = os.path.join(wd, 'analysis/predictions/ef_similarity_corplot.png')
    plt.savefig(image_path)
    plt.clf()


    # Resid /  similarity plot
    fig, axes = plt.subplots(1, 5, dpi=100, figsize=(14,7))
    for i in range(len(layers)):
        layer = layers[i]

        reg_df = pd.concat([pd.DataFrame(cors[layer]), pd.DataFrame(resids[layer])], axis=1)
        reg_df.columns = ['Similarity', 'Residuals']

        sns.regplot(x='Similarity', y='Residuals', data=reg_df, color='blue', ci=None, ax=axes[i], scatter_kws={'alpha':0.3})
        axes[i].set_title(f"{layer}")

        sns.set_style({"xtick.direction": "in","ytick.direction": "in", "font_scale": 15})
        sns.despine()

        # Correlate 
        r, p = pearsonr(reg_df['Residuals'], reg_df['Similarity'])
        print(f"{layer}")
        print(f'r = {r:.3f}\np = {p}\nr2 = {r**2:.3f}')

        axes[i].set_xlabel(f'r = {r:.3f}, p = {p:.3f}')
        axes[i].set_ylabel('')
        # axes[i].set_xticks([],size=35)
        # axes[i].set_yticks([], size=35)

    fig.supxlabel('Feature similarity')
    fig.supylabel('Absolute model residuals')
    image_path = os.path.join(wd, 'analysis/predictions/mod_similarity_corplot.png')
    plt.savefig(image_path)
    plt.clf()

# --------------- MAIN

# Extract features pretrained model
# pre_model = models.resnet50(pretrained=True)
# exp_features = extract(model = pre_model, m_type = 'pretrained')

# Extract features random model
# r_model = models.resnet50(pretrained=False)
# r_features = extract(model = r_model, m_type = 'random')

# Load features experimental images
exp_features = load_features(atype='exp')
random_features = load_features(atype='random')

# Preprocess behavioural data
bdata = preprocess_bdata()

# Set up random model
# PLS_model(bdata, random_features, exp=False, thres=0)

# Set up log regression model
# PLS_model(bdata, exp_features, exp=True, thres=0)

# Model per mask
model_per_mask()

# Calc feature distancese
# calc_distance(exp_features)
