# Data augmentation by reverberation with dynamic range compression

This is a extremely simple data augmentation toolkit 
based on Matlab (m-file) scripts and sox command

First we reverberate audio files, and next we can optionally also apply DRC to those audio files

## Folders
`m-scripts` contains m-files for reverb end with _en.m have the energy normalization step

There are there RIR folders, the three different group of rirs from downloaded from internet:
   -- MARDY
   -- TrainRIR
   -- 2003_estimated_ir

## How to use

To reverberate data (rir_label can be reverb, 2003, or mardy, see script for more info):
>nohup ./gen_RevDevelopment.sh parent_dir_of_folder_with_audio foder_generated_reverb_audio rir_label rir_waveform_location/ > RIR.log

>./gen_RevDevelopment.sh /nethome/hhu96/asc/ms_2020_subtask_a/DCASE/data_aug/ /nethome/hhu96/asc/ms_2020_subtask_a/DCASE/data_aug/rev_mardy/ mardy mardy_rirs/ 

 
To perform DRC on audio files, plese run `./drc.sh`. You can set RIR_DATASET to either "2003", "reverb", or "mardy" on row #60.
> ./drc.sh rev_mardy/audio/ audiorev_drec
