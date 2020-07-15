import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
from utils import *
from funcs import *
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import pandas as pd
import time

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# Define a helper function to evaluate the TFlite model using test dataset. 
def evaluate_model(interpreter, test_images, num_class):
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
        if (len(pred_output_all) - 1) % 1000 == 0:
            print('%d testing samples have been processed' % (len(pred_output_all) - 1))
        digit = np.argmax(output[0])
        prediction_digits.append(digit)
        
    pred_output_all = pred_output_all[1:,:]
    
    return pred_output_all, prediction_digits


num_freq_bin = 128 # number of the frequency bins
num_classes = 3    # number of the classes, for DCASE2020 1b, num_classes is 3
NumSamples = 8640 # number of the evluation samples


# PART I: train the fusion system with the development set
print('PART I: train the fusion system with the development set')

val_csv = 'data_2020/evaluate_setup/fold1_evaluate.csv'   # path to the validation subset
feat_path =  'features/logmel128_scaled_d_dd/'  # path to the extracted features
model_path_A = '../pretrained_models/smallfcnn-model-0.9618.hdf5'  # path to the pre-trained model A
model_path_B = '../pretrained_models/mobnet-model-0.9517.hdf5'  # path to the pre-trained model B

data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')

best_model_A = tf.keras.models.load_model(model_path_A)
best_model_B = tf.keras.models.load_model(model_path_B)

preds_A = best_model_A.predict(data_val)
preds_B = best_model_B.predict(data_val)

print('Learn the fusion system ... ')
# Stack predictions in to [rows, probabilities, members]
# Predictions from pre-trained sub-models are used as the training data for 
# meta-learner in the stacking ensemble
stackX = np.dstack((preds_A, preds_B))

# flatten predictions to [rows, probabilities * members]
stackedX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))

# fit the meta-learner by applying logistic regression
# the predictions on the validation set are used to train the fusion system
model_meta_learner = LogisticRegression()
model_meta_learner.fit(stackedX, y_val)

# PART II: apply the fusion system on the evaluation set.
print('PART II: apply the fusion system on the evaluation set.')

test_output_path = 'saved-model-fusion.csv'     # path to dump the results
model_path_A_quantized = '../pretrained_models/smallfcnn-model-0.9618-quantized.tflite' # path to the quantized model A
model_path_B_quantized = '../pretrained_models/mobnet-model-0.9517-quantized.tflite' # path to the quantized model B

data_eval = load_data_2020_eval(feat_path, val_csv, num_freq_bin, 'logmel')

# Load the model A into an interpreter
interpreter_quant_A = tf.lite.Interpreter(model_path=model_path_A_quantized)
interpreter_quant_A.allocate_tensors()

# Load the model B into an interpreter
interpreter_quant_B = tf.lite.Interpreter(model_path=model_path_B_quantized)
interpreter_quant_B.allocate_tensors()

# get the results of the quantized sfcnn model
preds_A, preds_class_idx = evaluate_model(interpreter_quant_A, 
                                    data_eval,
                                    num_class=num_classes)

# get the results of the quantized mobnetv2 model
preds_B, preds_class_idx = evaluate_model(interpreter_quant_B, 
                                    data_eval,  
                                    num_class=num_classes)

# get the results of the quantized model by fusion
stackX = np.dstack((preds_A, preds_B))
stackedX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
preds_meta = model_meta_learner.predict_proba(stackedX)

print('The following are the results of the fusion by applying stacking ensemble')
test_output_df = get_output_dic(preds_meta, NumSamples)

print('Saving the output csv file......')
test_output_df.to_csv(test_output_path, index=False, float_format='%.2f', sep=' ')
print('Output csv file has been saved successfully.')
print('ALL DONE !!!')


