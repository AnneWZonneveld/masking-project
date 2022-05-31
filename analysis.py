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
import joblib
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


def load_features(atype = 'exp'): # add type argument
    layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']

    features = {}
    if atype == 'exp':
        output_dir = os.path.join(wd, f'analysis/exp_features/')
    elif atype == 'animacy':
        output_dir = os.path.join(wd, f'analysis/animacy_features/')

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


def logit_model(data, features):
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from sklearn.metrics import f1_score, classification_report
    from sklearn.model_selection import RepeatedKFold, StratifiedKFold, GridSearchCV, cross_validate, train_test_split
    import statsmodels.api as sm
    from sklearn import preprocessing

    file_dir = os.path.join(wd, 'analysis', 'image_paths_exp.csv') # should be cc1
    image_paths = pd.read_csv(file_dir)['path'].tolist()
    concepts = pd.unique(concept_selection['concept']).tolist()

    n_trials = len(trial_file)
    trial_df = pd.DataFrame()
    # single_df = pd.DataFrame()
   
    no_mask = [i for i in range(len(trial_file)) if trial_file.iloc[i]['mask_type']=='no_mask']
    n_mask_trials = n_trials - len(no_mask)
    target_activations = np.zeros((n_mask_trials, n_components)) # later do more layers
    mask_activations = np.zeros((n_mask_trials, n_components))

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
            target_activation = features['layer5'][target_index, :]
            target_activations[no_mask_id, :] = target_activation
        
            # get mask path
            mask_path = os.path.join(wd, mask_path[53:])

            # get according activation
            mask_index = image_paths.index(mask_path)
            mask_activation = features['layer5'][target_index, :]
            mask_activations[no_mask_id, :] = mask_activation

            # get response for all participants (average?)
            responses = data.groupby(['index'])['answer']
            responses = data[data['index'] == trial_id]['answer'].tolist() #anwer or correct?
            subject_nrs = data[data['index'] == trial_id]['subject_nr'].tolist()
            mask_type = mask_path.split('/')[-3]

            # tmp_single['trialID'] = trial_id
            # tmp_single['target_path'] = target_path
            # tmp_single['mask_path'] = mask_path
            # tmp_single['mask_activation'] = mask_activation
            # tmp_single['target_activation'] = target_activation

            tmp['index'] = [trial_id for i in range(len(responses))]
            tmp['response'] = responses
            tmp['subject_nr'] = subject_nrs
            tmp['target_path'] = [target_path for i in range(len(responses))]
            tmp['mask_path'] = [mask_path for i in range(len(responses))]
            tmp['mask_type'] = [mask_type for i in range(len(responses))]
            tmp['mask_activation'] = [mask_activation for i in range(len(responses))]
            tmp['target_activation'] = [target_activation for i in range(len(responses))]

            trial_df = pd.concat([trial_df, tmp], ignore_index=True)
            # single_df = pd.concat([single_df, tmp_single], ignore_index=True)

            no_mask_id =+ 1

    shell()

    # Activations for all trials, all ppn
    X1 = np.zeros((len(trial_df), target_activation.shape[0]))
    X2 = np.zeros((len(trial_df), mask_activation.shape[0]))

    for i in range(len(trial_df)):
        X1[i, :] = trial_df['target_activation'].iloc[i]
        X2[i, :] = trial_df['mask_activation'].iloc[i]
    X = np.concatenate((X1, X2), axis=1)
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Activations for unique trials
    X_single = np.concatenate((target_activations, mask_activations), axis=1)
    scaler = preprocessing.StandardScaler().fit(X_single)
    X_single_scaled = scaler.transform(X_single)

    y = np.asarray(trial_df['response'])
    
    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    scale = preprocessing.StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test) 

    # Check for imbalance
    pd.Series(y_train).value_counts(normalize=True) # --> 0: 0.86, 1:0.14

    lr_basemodel = LogisticRegression(max_iter=5000, class_weight={0:0.14, 1:0.86})
    lr_basemodel.fit(X_train, y_train)

    # Train and test metrics
    print(f"Train accuracy: {lr_basemodel.score(X_train, y_train)}")
    print(f"Test accuracy: {lr_basemodel.score(X_test, y_test)}")
    y_train_pred = lr_basemodel.predict(X_train)
    y_test_pred = lr_basemodel.predict(X_test)
    print(f"F1 train score: {f1_score(y_train,y_train_pred)}")
    print(f"F1 test score: {f1_score(y_test,y_test_pred)}")
    print(f"Classification report:")
    print(f"{classification_report(y_test, y_test_pred)}")

    # Hyperparameter tuning
    lr=LogisticRegression(max_iter=5000)
    weights = np.linspace(0.0,0.99,2)
    param= {'C': [0.1, 0.5, 1,10,15,20], 'penalty': ['l1', 'l2'],"class_weight":[{0:x ,1:1.0 -x} for x in weights]}
    folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    
    # Gridsearch 
    grid_model= GridSearchCV(estimator= lr,param_grid=param,scoring="f1",cv=folds,return_train_score=True)
    grid_model.fit(X_train,y_train)
    print(f"GridSearch best F1 score: {grid_model.best_score}")
    best_params = grid_model.best_params
    print(f"GridSearch best parameters score: {best_params}")

    # Refit 
    lr2=LogisticRegression(class_weight={0:0.27,1:0.73},C=20,penalty="l2")
    lr2.fit(X_train,y_train)

    lr2_pred = lr2_pred.predict(X_test)
    print(f"Classification report:")
    print(f"{classification_report(y_test, lr2_pred)}")
    

    # Calculate noise ceiling
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

# Preprocess behavioural data
bdata = preprocess_bdata()

# Extract features all experimental images
exp_features = load_features(atype='exp')

# Set up log regression model
logit_model(bdata, exp_features)
