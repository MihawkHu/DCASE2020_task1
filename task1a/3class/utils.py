import numpy as np
import pandas as pd
import pickle


def load_data_2020(feat_path, csv_path, feat_dim, file_type):
    with open(csv_path, 'r') as text_file:
        lines = text_file.read().split('\n')
        for idx, elem in enumerate(lines):
            lines[idx] = lines[idx].split('\t')
            lines[idx][0] = lines[idx][0].split('/')[-1].split('.')[0]

        # remove first line
        lines = lines[1:]
        lines = [elem for elem in lines if elem != ['']]
        for idx, elem in enumerate(lines):
            lines[idx][-1] = lines[idx][-1].split('\r')[0]
        label_info = np.array(lines)
        
        data_df = pd.read_csv(csv_path, sep='\t', encoding='ASCII')
        ClassNames = np.unique(data_df['scene_label'])
        labels = data_df['scene_label'].astype('category').cat.codes.values

        feat_mtx = []
        for [filename, labnel] in label_info:
            filepath = feat_path + '/' + filename + '.logmel' 
            with open(filepath,'rb') as f:
                temp=pickle.load(f, encoding='latin1')
                feat_mtx.append(temp['feat_data'])

        feat_mtx = np.array(feat_mtx)

        return feat_mtx, labels


def load_data_2020_splitted(feat_path, csv_path, feat_dim, idxlines, file_type):
    with open(csv_path, 'r') as text_file:
        lines = text_file.read().split('\n')
        for idx, elem in enumerate(lines):
            lines[idx] = lines[idx].split('\t')
            lines[idx][0] = lines[idx][0].split('/')[-1].split('.')[0]

        # remove first line
        lines = lines[1:]
        lines = [lines[i] for i in idxlines]
        lines = [elem for elem in lines if elem != ['']]
        for idx, elem in enumerate(lines):
            lines[idx][-1] = lines[idx][-1].split('\r')[0]
        label_info = np.array(lines)
        
        data_df = pd.read_csv(csv_path, sep='\t', encoding='ASCII')
        ClassNames = np.unique(data_df['scene_label'])
        labels = data_df['scene_label'].astype('category').cat.codes.values
        labels = [labels[i] for i in idxlines]
        
        feat_mtx = []
        for [filename, labnel] in label_info:
            filepath = feat_path + '/' + filename + '.logmel'
            with open(filepath,'rb') as f:
                temp=pickle.load(f, encoding='latin1')
                feat_mtx.append(temp['feat_data'])

        feat_mtx = np.array(feat_mtx)

        return feat_mtx, labels

def load_data_2020_test(feat_path, csv_path, csv_path_true, feat_dim, file_type):
    with open(csv_path_true, 'r') as text_file:
        lines = text_file.read().split('\n')
        for idx, elem in enumerate(lines):
            lines[idx] = lines[idx].split('\t')
            lines[idx][0] = lines[idx][0].split('/')[-1].split('.')[0]

        # remove first line
        lines = lines[1:]
        lines = [elem for elem in lines if elem != ['']]
        for idx, elem in enumerate(lines):
            lines[idx][-1] = lines[idx][-1].split('\r')[0]
        label_info_true = np.array(lines)

    with open(csv_path, 'r') as text_file:
        lines = text_file.read().split('\n')
        for idx, elem in enumerate(lines):
            lines[idx] = lines[idx].split('\t')
            lines[idx][0] = lines[idx][0].split('/')[-1].split('.')[0]

        # remove first line
        lines = lines[1:]
        lines = [elem for elem in lines if elem != ['']]
        for idx, elem in enumerate(lines):
            lines[idx][-1] = lines[idx][-1].split('\r')[0]
        label_info = np.array(lines)

        data_df = pd.read_csv(csv_path, sep='\t', encoding='ASCII')
        ClassNames = np.unique(data_df['scene_label'])
        labels = data_df['scene_label'].astype('category').cat.codes.values

        feat_mtx = []
        for [filename, label] in label_info:
            print(filename)
            filepath = feat_path + '/' + label_info_true[int(filename)][0] + '.logmel'
            print(filepath)
            with open(filepath,'rb') as f:
                temp=pickle.load(f, encoding='latin1')
                feat_mtx.append(temp['feat_data'])

        feat_mtx = np.array(feat_mtx)
        return feat_mtx, labels


def deltas(X_in):
    X_out = (X_in[:,:,2:,:]-X_in[:,:,:-2,:])/10.0
    X_out = X_out[:,:,1:-1,:]+(X_in[:,:,4:,:]-X_in[:,:,:-4,:])/5.0
    return X_out
