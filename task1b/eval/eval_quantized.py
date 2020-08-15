#### For testing the TF Lite model, which is saved as .tflite format

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
from utils import *
from funcs import *
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
import pandas as pd

num_freq_bin = 128
num_classes = 3

val_csv = 'data_2020/evaluation_setup/fold1_evaluate.csv'
feat_path = 'features/logmel128_scaled_d_dd/'
model_path = '../pretrained_models/smallfcnn-model-0.9618-quantized.tflite'
data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes)
print(data_val.shape)
print(y_val.shape)

# Load the model into an interpreter
interpreter_quant = tf.lite.Interpreter(model_path=model_path)
interpreter_quant.allocate_tensors()

# Evaluate the models
## Define a helper function to evaluate the TFlite model using test dataset. 
def evaluate_model(interpreter, test_images, test_labels, num_class, is_eval = False):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    
    # Run predictions on every image in the test dataset.
    prediction_digits = []
    pred_output_all = np.empty([1,num_class])
    for test_image in test_images:
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)
        
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_index)
        pred_output = output[0]
        pred_output.reshape([1,num_class])
        pred_output_all = np.vstack((pred_output_all, pred_output))
        digit = np.argmax(output[0])
        prediction_digits.append(digit)
        
        
    pred_output_all = pred_output_all[1:,:]
    
    if is_eval:
        return pred_output_all, prediction_digits
    else:
        # Compare prediction results with ground truth labels from the validation set to calculate accuracy.
        accurate_count = 0
        for index in range(len(prediction_digits)):
            if prediction_digits[index] == test_labels[index]:
                accurate_count += 1
        accuracy = accurate_count * 1.0 / len(prediction_digits)
    
        return accuracy, pred_output_all, prediction_digits
    

'''
# This is used for testing parts of the testing set.
test_slide_len = len(y_val)
overall_acc, preds, preds_class_idx = evaluate_model(interpreter_quant, 
                                    data_val[0:test_slide_len,:,:,:], y_val[0:test_slide_len,], 
                                    # output_shape=[test_slide_len,num_classes], 
                                    num_class=num_classes)
over_loss = log_loss(y_val_onehot, preds)
'''

overall_acc, preds, preds_class_idx = evaluate_model(interpreter_quant, 
                                    data_val, y_val, 
                                    num_class=num_classes)
over_loss = log_loss(y_val_onehot, preds)

# Map the class index (0,1,2) to class name (indoor, outdoor, transportation).
preds_class_idx = np.array(preds_class_idx)
preds_class_idx = np.reshape(preds_class_idx, (len(y_val),1))
preds_class = np.where(preds_class_idx > 1, 'transportation', 
                       (np.where(preds_class_idx < 1, 'indoor', 'outdoor')))

# Output the prediction resutls to a csv file with a format required by the DCASE.
test_output_df = pd.DataFrame({
                               'scene_label': preds_class[:,0],
                               'indoor': preds[:,0],
                               'outdoor': preds[:,1],
                               'transportation': preds[:,2]})
                               
test_output_df.to_csv('result.csv', index=False, float_format='%.2f')
print('Output csv file has been saved successfully.')


np.set_printoptions(precision=3)
print("\n\nVal acc: ", "{0:.3f}".format(overall_acc))
print("Val log loss:", "{0:.3f}".format(over_loss))

y_pred_val = np.argmax(preds,axis=1)
conf_matrix = confusion_matrix(y_val,y_pred_val)
print("\n\nConfusion matrix:")
print(conf_matrix)
conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
recall_by_class = np.diagonal(conf_mat_norm_recall)
mean_recall = np.mean(recall_by_class)

dev_test_df = pd.read_csv(val_csv,sep='\t', encoding='ASCII')
ClassNames = np.unique(dev_test_df['scene_label'])

print("Class names:", ClassNames)
print("Per-class val acc: ",recall_by_class, "\n\n")











