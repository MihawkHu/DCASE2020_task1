#!/usr/bin/env bash


##
#
#
# E.g.: nohup ./gen_RevDevelopment.sh DCASE2020 DCASE/data_reverb 2003 2003_estimated_ir/ > RIR.log &
#                                     DCASE2020 is where you store the foder audio with the development audio files
#                                     data_reverb is the folder where the reverb data are dumped ( under the folder  audio )
#                                     2003_estimated_ir/ folder where the RIR waveforms are located (please use / at the end, I was lazy here )  
##

usage="nohup $0 input_audio_dir output_dir rir_dataset rir_dir > REVERB.log &"


if [ $# !=  4 ]; then
    echo "Example to use this script:"
    echo ${usage}
    exit 1;
fi

# My path for MARDY
#RIR="/nethome/hhu96/asc/ms_2020_subtask_a/DCASE/MARDY/"
# My path for 2003
#RIR="/nethome/hhu96/asc/ms_2020_subtask_a/DCASE/2003_estimated_ir/"
# My path for Reverb Challenge
#RIR="/nethome/hhu96/asc/ms_2020_subtask_a/DCASE/TrainRIR/"

indir=${1}
outdir=${2}
RIR_DATASET=${3}
RIR=${4}

mdir="m-scripts"

if [ ! -d ${indir} ]; then
        echo "input parameters must be directories"
        echo ${usage}
        exit 1
fi

#mkdir -p ${outdir} 

clean_up() {
    echo "# "
    echo "Something went wrong, housekeeping performed"
    echo ${usage}
    rm -r -f ${outdir}
    exit 1
}
trap clean_up SIGINT SIGTERM

## RIR_DATASET Selects the m-file to use
case ${RIR_DATASET} in
    2003 )   mname="Generate_mcTrainData2003_estimated_ir_dcase";;
    mardy )  mname="Generate_mcTrainDataMardy_dcase";;
    reverb)  mname="Generate_mcTrainData_dcase";;
    * ) echo "option not supported"; exit 1;;
esac

cp ${mdir}/${mname}.m . || exit 1; 



matlab -nodisplay -nodesktop -nojvm -r "${mname}('$1','$2','$RIR')" || clean_up

rm ${mname}.m
echo "Done!"

