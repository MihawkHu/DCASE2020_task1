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
            filepath = feat_path + '/' + filename + '.' + file_type
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
            filepath = feat_path + '/' + filename + '.' + file_type
            with open(filepath,'rb') as f:
                temp=pickle.load(f, encoding='latin1')
                feat_mtx.append(temp['feat_data'])

        feat_mtx = np.array(feat_mtx)

        return feat_mtx, labels

def load_data_2020_withaug_splitted(csv_path, feat_dim, idxlines, file_type):
    with open(csv_path, 'r') as text_file:
        lines = text_file.read().split('\n')
        for idx, elem in enumerate(lines):
            lines[idx] = lines[idx].split('\t')

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
        for [filename, label] in label_info:
            filepath = filename
            with open(filepath,'rb') as f:
                temp=pickle.load(f, encoding='latin1')
                feat_mtx.append(temp['feat_data'])

        feat_mtx = np.array(feat_mtx)

        return feat_mtx, labels


def deltas(X_in):
    X_out = (X_in[:,:,2:,:]-X_in[:,:,:-2,:])/10.0
    X_out = X_out[:,:,1:-1,:]+(X_in[:,:,4:,:]-X_in[:,:,:-4,:])/5.0
    return X_out


def get_output_dic(preds, num_samples):
    y_pred_val = np.argmax(preds,axis=1)
    # Save the predictions to csv files
    # Map the class index (0,1,2) to class name (indoor, outdoor, transportation).
    preds_class_idx = np.array(y_pred_val)
    preds_class_idx = np.reshape(preds_class_idx, (num_samples,1))
    preds_class = np.where(preds_class_idx > 1, 'transportation', 
                           (np.where(preds_class_idx < 1, 'indoor', 'outdoor')))

    # Output the prediction resutls to a csv file.
    test_output_df = pd.DataFrame({'scene_label': preds_class[:,0],
                                   'indoor': preds[:,0],
                                   'outdoor': preds[:,1],
                                   'transportation': preds[:,2]})
                               
    return test_output_df
