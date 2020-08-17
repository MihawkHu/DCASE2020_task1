### Data augmentation by audio based methods

Four methods are coverred in this foloder:  
* change speed (time)
* shift pitch (pitch)
* add random noise (noise)
* mix audio files from the same class (add)  

#### How to use
To generate augmented features, please use `gen_extr_feat_2020_*.py`.   

The given four different scripts will load, augment, and extrat features of original data. Each time runnning will generate one time (~13k) of original data.  

The csv file for each kind of data is given as *.csv in the folder `./csv_files`.
