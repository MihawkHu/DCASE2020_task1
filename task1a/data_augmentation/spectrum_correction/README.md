### Data augmentation by spectrum correction method

Spectrum correction is proposed in [Acoustic Scene Classification for Mismatched Recording Devices Using Heated-Up Softmax and Spectrum Correction](https://ieeexplore.ieee.org/document/9053582),  and demonstrated moderate device  adaptation  properties.   However,  spectrum correction aims at transforming a given input spectrum to that of a reference, possibly ideal, device.  Different from the original idea, we here employ spectrum correction as a data augmentation technique. To this end, we had to modify the originalprocedure as follows:  
* Step 1: we create a reference device spec-trum, by averaging the spectrum from all training devices ex-cept that from device A; 
* Step 2: we correct the spectrum of eachtraining waveform collected with device A to obtain extra data.

#### How to use
Use `spec_correction_together.py` to generate extra data

Be carefully for each file address in the py file

`fold1_train_speccorr_together.csv` is the csv file for generated data. 
