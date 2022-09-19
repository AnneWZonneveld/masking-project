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
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import make_pipeline, Pipeline
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import spearmanr, ttest_1samp, pearsonr, ttest_rel, wilcoxon

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


def logit_model(data, exp_features, random_features):

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
                target_activation = exp_features[layer][target_index, :]
                mask_activation = exp_features[layer][target_index, :]
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
    
    shell()

    all_X = np.concatenate((X['layer1'], X['layer2'], X['layer3'], X['layer4'], X['layer5']), axis=1)
    X['all'] = all_X
    file_path = os.path.join(wd, 'analysis/stats/X')
    file = open(file_path, 'wb')
    pkl.dump(X, file)
    file.close()

    # Mask / no mask mean accuracy 
    type_means = trial_df.groupby(['mask_type'])['correct'].mean()
    
    trial_means = trial_df.groupby(['index'])['correct'].mean()
    mask_types = trial_file['mask_type']
    GA_df = pd.concat([trial_means, mask_types], ignore_index=True, axis=1)
    GA_df = GA_df.rename(columns={0: "Accuracy", 1: "Mask type"})

    # Plot accuracies 
    masks_ordered = ['no_mask', '2_scrambled', '5_lines', '6_blocked', '4_geometric', '1_natural']
    GA_df["Mask type"] = pd.Categorical(GA_df["Mask type"], masks_ordered)
    sns.set_palette('tab10')
    sns.set_style("white")
    g = sns.boxplot(data=GA_df, x="Mask type", y="Accuracy", showmeans=True, meanprops={"markeredgecolor" : "black", "markerfacecolor":"white"})
    plt.title("Accuracy per unique valid trials")
    g.set_xticklabels(['no mask', 'scrambled', 'lines', 'blocked', 'geometric', 'natural'])

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    image_paths = os.path.join(wd, 'analysis/Accuracy-per-mask.png')
    plt.savefig(image_paths)
    plt.clf()

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
    masks_ordered = ['1_natural', '5_lines', '6_blocked', '4_geometric', '2_scrambled']
    GA_df_masks["Mask type"] = pd.Categorical(GA_df_masks["Mask type"], masks_ordered)
    g = sns.boxplot(data=GA_df_masks, x="Mask type", y="Efficacy", showmeans=True, meanprops={"markeredgecolor" : "black", "markerfacecolor":"white"})
    plt.title("Mask efficacy per unique trial")
    plt.ylim([-0.1, 0.8])
    sns.despine(offset=10)
    g.set_xticklabels(['natural', 'lines', 'blocked', 'geometric','scrambled'])
    plt.tight_layout()
    image_path = os.path.join(wd, 'analysis/Efficacy-per-mask.png')
    plt.savefig(image_path)
    plt.clf()

    # Model predictions per layer
    lr = LinearRegression()
    clf = make_pipeline(VarianceThreshold(), preprocessing.StandardScaler(), lr)

    y = np.asarray(GA_df_masks['Efficacy'])
    preds = np.zeros((n_mask_trials, n_layers + 1))
    for i in range(n_layers):
        layer = layers[i]
        print(f"CV for {layer}")
        
        preds[:, i] = cross_val_predict(clf, X[layer], y, cv=n_mask_trials)
    print(f"CV all layer model")
    preds[:, i + 1] = cross_val_predict(clf, X['all'], y, cv=n_mask_trials)
    
    file_path = os.path.join(wd, 'analysis/predictions/')
    if not os.path.exists(file_path):
            os.makedirs(file_path)

    np.save(os.path.join(file_path, "lr_predictions.npy"), preds)

    # MAE error - per layer
    preds = glob.glob(os.path.join(file_path, 'lr_predictions.npy'))[0]
    preds = np.load(preds)

    MAEs = []
    for i in range(n_layers + 1):
        print(f'Layer {i}')

        y_pred = preds[:, i]
        error = MAE(y, y_pred)
        MAEs.append(error)
    
    layers_oi = layers + ['all']
    MAE_df = pd.concat([pd.DataFrame(MAEs), pd.DataFrame(layers_oi)], axis=1)
    MAE_df.columns = ['MAE', 'layer']

    # Save 
    file_path = os.path.join(wd, 'analysis/stats/')
    if not os.path.exists(file_path):
            os.makedirs(file_path)
    MAE_df.to_csv(os.path.join(file_path, 'MAE_per_layer.csv'), index=False)

    # MAE plot
    sns.pointplot(data=MAE_df, x="layer", y="MAE")
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    image_path = os.path.join(wd, 'analysis/MAEs-layers.png')
    plt.savefig(image_path)

    # Evaluate MAE per mask type
    MAE_mask_df = pd.DataFrame()
    mask_MAEs = []
    layer_nr = layers_oi * len(masks_ordered)
    mask_nr = []

    for j in range(len(masks_ordered)):

        mask_type = masks_ordered[j]
        mask_name = mask_type.split("_")[1]
        mask_ids = np.array(GA_df_masks[GA_df_masks['Mask type']== mask_type].index)

        for i in range(len(layers_oi)):

            layer = layers_oi[i]

            mask_pred = preds[mask_ids, i]
            y_mask = y[mask_ids]

            error = MAE(y_mask, mask_pred)
            mask_MAEs.append(error)

            mask_nr.append(mask_name)
    
    MAE_mask_df['MAE'] = mask_MAEs
    MAE_mask_df['Mask'] = mask_nr
    MAE_mask_df['Layer'] = layer_nr

    # Save 
    file_path = os.path.join(wd, 'analysis/stats/')
    if not os.path.exists(file_path):
            os.makedirs(file_path)
    MAE_mask_df.to_csv(os.path.join(file_path, 'MAE_per_mask.csv'), index=False)

    # Plot MAE per mask per layer
    sns.pointplot(data=MAE_mask_df, x='Layer', y='MAE', hue='Mask')
    sns.despine(offset=10)
    plt.ylim([0.07, 0.14])
    xlabels = ['layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5', 'all']
    plt.xticks(np.arange(len(xlabels)),xlabels)
    plt.xlabel('')
    plt.tight_layout()
    image_path = os.path.join(wd, 'analysis/MAEs-per-mask.png')
    plt.savefig(image_path)
    plt.clf()

    # Plot MAE per mask per layer - swapped
    sns.pointplot(data=MAE_mask_df, x='Mask', y='MAE', hue='Layer')
    sns.despine(offset=10)
    plt.ylim([0.07, 0.14])
    plt.tight_layout()
    image_path = os.path.join(wd, 'analysis/MAEs-per-mask-alt.png')
    plt.savefig(image_path)
    plt.clf()


    # Prediction per image for best layer (= layer4)
    tmp_df_1 = pd.DataFrame(preds[:, 3]).reset_index()
    tmp_df_2 = pd.DataFrame(y).reset_index()

    sns.scatterplot(x='index', y=0, data=tmp_df_2, markers='d', color ='b', alpha=0.6)
    sns.scatterplot(x='index', y=0, data=tmp_df_1, markers='d', color ='r')

    sns.despine(offset=10)
    plt.xticks([])
    plt.xlabel('')
    plt.ylabel('Mask efficacy')

    handles = [Line2D([0], [0], color='blue', lw=3, linestyle='-'), Line2D([0], [0], color='r', lw=3, linestyle='-'),]
    labels =[ 'observed mask efficacy', 'layer 4 prediction']

    l = plt.legend(handles, labels, loc=1,
                borderaxespad=0., frameon=False)

    plt.tight_layout()
    image_path = os.path.join(wd, 'analysis/prediction-per-image.png')
    plt.savefig(image_path)

    # Prediction per image for best layer residuals (= layer4)
    resid  = y - y_pred
    tmp_df = pd.DataFrame(resid).reset_index()

    sns.scatterplot(x='index', y=0, data=tmp_df, markers='d', color ='b', alpha=0.6)
    sns.despine(offset=10)
    plt.xticks([])
    plt.ylabel('Residuals')

    plt.tight_layout()
    image_path = os.path.join(wd, 'analysis/residuals-per-image.png')
    plt.savefig(image_path)


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
        r, p = pearsonr(pred_df['y'], pred_df['y_pred'])
        print(f"{layer}")
        print(f'r = {r:.3f}\np = {p}\nr2 = {r**2:.3f}')

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
    lr = LinearRegression()
    clf = make_pipeline(VarianceThreshold(), preprocessing.StandardScaler(), lr)

    n_mask_trials = len(GA_df_masks)    
    masks_ordered = ['1_natural', '5_lines', '6_blocked', '4_geometric', '2_scrambled']
    layers_oi = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'all']

    preds = np.zeros((n_mask_trials/len(masks_ordered), len(layers_oi)))
    MAE_masks = np.zeros((len(layers_oi), len(masks_ordered)))

    for i in range(len(layers_oi)):
        layer = layers_oi[i]
        print(f" {layer}")

        for j in range(len(masks_ordered)):
            mask_type = masks_ordered[j]
            print(f" {mask_type}")

            mask_ids = np.array(GA_df_masks[GA_df_masks['Mask type']== mask_type].index)
            X_select = X[layer][mask_ids]
            y_select= y[mask_ids]

            mask_pred = cross_val_predict(clf, X_select, y_select, cv=X_select.shape[0])
            preds[:, i] = mask_pred


            clf.fit(X_train, y_train)
            mask_pred = clf.predict(X_test)
            error = MAE(y_test, mask_pred)
            MAE_masks[i, j] = error
    
    file_path = os.path.join(wd, 'analysis/predictions/')
    if not os.path.exists(file_path):
            os.makedirs(file_path)

    np.save(os.path.join(file_path, "MAE_per_mask.npy"), preds)

    # Plot MAE per mask per layer
    palette = sns.color_palette("tab10", len(masks_ordered))
    for i in range(len(masks_ordered)):
        mask = masks_ordered[i]
        tmp_df = pd.DataFrame(MAE_masks[:, i]).reset_index()
        tmp_df = tmp_df.rename(columns = {'index': 'layer', 0:'MAE'})
        sns.pointplot(data=tmp_df, x="layer", y="MAE", color=palette[i], label=mask)

    sns.despine(offset=10)
    # plt.ylim([0.07, 0.14])
    xlabels = ['layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5', 'all']
    plt.xticks(np.arange(len(xlabels)),xlabels)
    plt.xlabel('')

    handles = []
    for colour in palette:
        handles.append(Line2D([0], [0], color=colour, lw=3, linestyle='-'))
    # labels = ['natural', 'lines', 'blocked', 'geometric', 'scrambled']
    
    l = plt.legend(handles, labels, loc=2,
                borderaxespad=0., frameon=False)

    plt.tight_layout()
    image_path = os.path.join(wd, 'analysis/MAEs-fitted-per-mask.png')
    plt.savefig(image_path)
    plt.clf()


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

# Set up log regression model
# logit_model(bdata, exp_features, random_features)

# Model per mask
model_per_mask()


# Distance analysis
# distance_analysis(bdata, exp_features, random_features)