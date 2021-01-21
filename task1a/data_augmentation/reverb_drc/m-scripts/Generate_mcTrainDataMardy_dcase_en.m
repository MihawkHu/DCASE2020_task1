function Generate_mcTrainDataMardy_dcase(DCASE_dir_name, save_dir, path_rir)
%
% Input variables:
%    DCASE_dir_name: string name of user's clean wsjcam0 corpus directory 
%                  (*Directory structure for wsjcam0 corpushas to be kept as it is after obtaining it from LDC. 
%                    Otherwise this script does not work.)
%
% This function generates multi-condition traiing data, and it is adapted from REVERB 2014 Challenge
% based on the following items:
%  1. DCASE corpus (distributed from the DCASE)
%  2. room impulse responses (ones under ./RIR/)
% Generated data has the same directory structure as original DCASE corpus. 
%

if nargin<2
   error('Usage: Generate_mcTrainData_dcase(DCASE_data_path, save_dir)  *Note that the input variable DCASE_data_path should indicate the directory name of your clean WSJCAM0 corpus. '); 
end
if exist([DCASE_dir_name,'/audio/'])==0
   error(['Could not find dcase corpus : Please confirm if ',DCASE_dir_name,' is a correct path to your clean DCASE corpus']); 
end

if ~exist('save_dir', 'var')
    error('You have to set the save_dir variable in the code before running this script!')
end

display(['Name of directory for original DCASE: ',DCASE_dir_name])
display(['Name of directory to save generated multi-condition training data: ',save_dir])


% list of training data from device A
flist1='evaluation_setup/fold1_train_a.lst';

%
% List of RIRs
%
%RIR_Path='/nethome/hhu96/asc/ms_2020_subtask_a/DCASE/MARDY/';
RIR_Path=path_rir;
num_RIRvar=9;
RIR_sim1=strcat(RIR_Path,'ir_1_C_1.wav'); 
RIR_sim2=strcat(RIR_Path,'ir_2_C_1.wav'); 
RIR_sim3=strcat(RIR_Path,'ir_3_C_1.wav');  
RIR_sim4=strcat(RIR_Path,'ir_1_L_1.wav');  
RIR_sim5=strcat(RIR_Path,'ir_2_L_1.wav');
RIR_sim6=strcat(RIR_Path,'ir_3_L_1.wav');
RIR_sim7=strcat(RIR_Path,'ir_1_R_1.wav'); 
RIR_sim8=strcat(RIR_Path,'ir_2_R_1.wav'); 
RIR_sim9=strcat(RIR_Path,'ir_3_R_1.wav'); 

%
% Start generating noisy reverberant data with creating new directories
%

fcount=1;
rcount=1;

if save_dir(end)=='/';
    save_dir_tr=[save_dir,''];
else
    save_dir_tr=[save_dir,'/'];
end
mkdir([save_dir_tr]);
%mkdir([save_dir,'/taskfiles/'])


prev_fname='dummy';

for nlist=1:1
    % Open file list
    eval(['fid=fopen(flist',num2str(nlist),',''r'');']);

    while 1
        
        % Set data file name
        fname=fgetl(fid);
        if ~ischar(fname);
            break;
        end
        
        idx1=find(fname=='/');  
        
        % Make directory if there isn't any
        if ~strcmp(prev_fname,fname(1:idx1(end)))
            mkdir([save_dir_tr fname(1:idx1(end))])
        end
        prev_fname=fname(1:idx1(end));
       
        % load speech signal 
        [x, fs] = audioread([DCASE_dir_name,fname,'.wav']);
        energy0 = sum(x.^2);
        
        % load RIR and noise for "THIS" utterance
        eval(['[RIR, rfs]=audioread(RIR_sim',num2str(rcount),');']);

        [p, q] = rat(fs/rfs);
        RIR = resample(RIR(:,1), p, q);
        % Generate 1ch reverberant data        
        y=gen_obs(x,RIR);


        % cut to length of original signal
        y = y(1:size(x,1),:);
        energy1 = sum(y.^2);
        y = y * sqrt(energy0/energy1);

        % rotine to cyclicly switch RIRs, utterance by utterance 
        rcount=rcount+1;
        if rcount>num_RIRvar;rcount=1;end
        

        % save the data

        %y=y/4; % common normalization to all the data to prevent clipping
               % denominator was decided experimentally
        % check clipping and act accordingly
       % if (max(abs(y))>=1.0)
       %     display(['max(abs(y))',num2str(max(abs(y))),' scaled'])
       %     max_s=max(y);
       %     min_s=min(y);
       %     max_t=0.9;
       %     min_t=-0.9;
       %     y = ((max_t-min_t)/(max_s-min_s))*(y-min_s)+min_t;
       %     %y = y / (max(abs(y))*1.001);
       %     display(['max(abs(y))',num2str(max(abs(y))),' after scaled'])
       % end

        if(max(abs(y))>=1)
            display(['max(abs(y))',num2str(max(abs(y))),' scaled'])
            y = (y / max(abs(y))) * (1-eps);
        end

        fout=[save_dir_tr,fname,'.wav'];
        audiowrite(fout,y,fs,'BitsPerSample',24);
           
        display(['sentence ',num2str(fcount),' finished! (Multi-condition training data)'])
        fcount=fcount+1;

    end
end

quit

%%%%
function [y]=gen_obs(x,RIR)
% function to generate noisy reverberant data

%x=x';

% calculate direct+early reflection signal for calculating SNR
[val,delay]=max(RIR(:,1));

% obtain reverberant speech
rev_y=fconv(x,RIR(:,1));

y = rev_y(delay:end,:);


%%%%
function [y]=fconv(x, h)
%FCONV Fast Convolution
%   [y] = FCONV(x, h) convolves x and h, and normalizes the output  
%         to +-1.
%
%      x = input vector
%      h = input vector
% 
%      See also CONV
%
%   NOTES:
%
%   1) I have a short article explaining what a convolution is.  It
%      is available at http://stevem.us/fconv.html.
%
%
%Version 1.0
%Coded by: Stephen G. McGovern, 2003-2004.
%
%Copyright (c) 2003, Stephen McGovern
%All rights reserved.
%
%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
%AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
%ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
%LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
%CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
%SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
%INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
%CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
%ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
%POSSIBILITY OF SUCH DAMAGE.

Ly=length(x)+length(h)-1;  % 
Ly2=pow2(nextpow2(Ly));    % Find smallest power of 2 that is > Ly
X=fft(x, Ly2);   % Fast Fourier transform
H=fft(h, Ly2);   % Fast Fourier transform
Y=X.*H;          % 
y=real(ifft(Y, Ly2));      % Inverse fast Fourier transform
y=y(1:1:Ly);               % Take just the first N elements
