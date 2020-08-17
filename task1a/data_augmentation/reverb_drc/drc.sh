#!/bin/bash
#
# A basic bash script to perform Dynamic Range Compression
# using sox. 
#
#   Examples of usage:
#    ./drc.sh data_2020/2003-set2/audio data_2020/2003-set2-drc/audio
#
#    data_2020/2003-set2/audio is the location of the reverberated audio files
#    data_2020/2003-set2-drc/audio is where the DRC audio files are dumped
#
#


####   RIR resources
#
# Mardy: https://www.commsp.ee.ic.ac.uk/~sap/resources/mardy-multichannel-acoustic-reverberation-database-at-york-database/
# 2003: ihttps://www.smard.es.aau.dk
# reverb challenge: https://reverb2014.dereverberation.com
#
#

usage="nohup $0 input_audio_dir output_dir > DRC.log &"
if [ $# !=  2 ]; then
    echo "Example to use this script:"
    echo ${usage}
    exit 1;
fi

indir=${1}
outdir=${2}

if [ ! -d ${indir} ]; then
        echo "input parameters must be directories"
        echo ${usage}
fi

tmpdir=drclog
mkdir -p ${tmpdir}
mkdir -p ${outdir} 

clean_up() {
    echo "# "
    echo "Something went wrong, housekeeping performed"
    echo ${usage}
    rm -r -f ${tmpdir}
    rm -r -f ${outdir}
    exit 1
}
trap clean_up SIGINT SIGTERM 

clean_tmp() {
    echo "Removing temporary dir and files"
    rm -r -f ${tmpdir}
}

infiles=$(ls ${indir})

# Set RIR dataset used  among 2003, reverb, and mardy
RIR_DATASET="mardy"

## set compand parameters to avoid clipping when applying drc - different values for different RIRs
case ${RIR_DATASET} in
    2003 )   cpd1="compand 0.2,0.3 6:-90,-60,-60,-20,0,0 2 -90";
             cpd2="compand 0.2,0.3 6:-90,-60,-40,-20,0,0 2 -90";
             cpd3="compand 0.2,0.3 6:-90,-70,-50,-20,0,0 5 -90";
             cpd4="compand 0.2,0.3 6:-90,-60,-60,-20,0,0 3 -90";
             cpddef="compand 0.2,0.3 6:-90,-60,-60,-20,0,0 -5 -90";;
    mardy )  cpd1="compand 0.2,0.3 6:-90,-70,-60,-40,0,0 2 -90";
             cpd2="compand 0.2,0.3 6:-70,-60,-20,-10,0,0 2 -90";
             cpd3="compand 0.2,0.3 6:-90,-70,-35,-20,0,0 5 -90";
             cpd4="compand 0.2,0.3 6:-90,-60,-60,-20,0,0 1 -90";
             cpd5="compand 0.2,0.3 6:-80,-50,-50,-20,0,0 1 -90";
             cpddef="compand 0.2,0.3 6:-90,-80,-30,-20,0,0 -3 -90";;
    reverb)  cpd1="compand 0.2,0.3 6:-90,-85,-70,-60,-30,-20,0,0 2 -90";
             cpd2="compand 0.2,0.3 6:-90,-75,-40,-25,0,0 2 -90";
             cpd3="compand 0.2,0.3 6:-90,-90,-60,-50,-20,-20,0,0 4 -90";
             cpd4="compand 0.2,0.3 6:-70,-60,-60,-40,-20,-10,0,0 3 -90";
             cpddef="compand 0.2,0.3 6:-90,-90,-70,-65,-30,-25,0,0 -6 -90";;
    * ) echo "Unavailable option";
         exit 1;;

esac

echo "DRC with ${RIR_DATASET} reverb audio data";


ccount=param1
echo ${ccount}
for line in ${infiles}; do
    fname=${line##*\.}
    [ ${fname} = "wav" ] || exit 1; 
    infile=${indir}/${line}
    outfile=${outdir}/${line}
    case ${ccount} in
        param1 ) echo -e "sox ${infile} ${outfile} ${cpd1}";
                sox ${infile} ${outfile} ${cpd1} 2> drclog/tmp.log;
                    numline=$(wc -l drclog/tmp.log | awk '{print $1}');
                if [ ${numline} != "0" ];then 
                    echo -e "redo \n sox ${infile} ${outfile} ${cpddef}";
                    sox ${infile} ${outfile} ${cpddef}; 
                fi
                ccount=param2;;
        param2 ) echo -e "sox ${infile} ${outfile} ${cpd2}";
                sox ${infile} ${outfile} ${cpd2} 2> drclog/tmp.log;
                numline=$(wc -l drclog/tmp.log | awk '{print $1}');
                if [ ${numline} != "0" ];then 
                    echo -e "redo \n sox ${infile} ${outfile} ${cpddef}";
                    sox ${infile} ${outfile} ${cpddef}; 
                fi
                ccount=param3;;
        param3 ) echo -e "sox ${infile} ${outfile} ${cpd3}";
                sox ${infile} ${outfile} ${cpd3} 2> drclog/tmp.log;
                numline=$(wc -l drclog/tmp.log | awk '{print $1}');
                if [ ${numline} != "0" ];then 
                    echo -e "redo \nsox ${infile} ${outfile} ${cpddef}";
                    sox ${infile} ${outfile} ${cpddef}; 
                fi
                ccount=param4;;
        param4 ) echo -e "sox ${infile} ${outfile} ${cpd4}";
                sox ${infile} ${outfile} ${cpd4} 2> drclog/tmp.log;
                numline=$(wc -l drclog/tmp.log | awk '{print $1}');
                if [ ${numline} != "0" ];then 
                    echo -e "redo \n sox ${infile} ${outfile} ${cpddef}";
                    sox ${infile} ${outfile} ${cpddef}; 
                fi
                ccount=param5;;
        param5 ) echo -e "sox ${infile} ${outfile} ${cpd5}";
                sox ${infile} ${outfile} ${cpd5} 2> drclog/tmp.log;
                numline=$(wc -l drclog/tmp.log | awk '{print $1}');
                if [ ${numline} != "0" ];then 
                    echo -e "redo \n sox ${infile} ${outfile} ${cpddef}";
                    sox ${infile} ${outfile} ${cpddef}; 
                fi  
                ccount=param1;;
        * ) echo "unkown combination";
            exit 1;;
    esac
done

clean_tmp

echo "Done!"




