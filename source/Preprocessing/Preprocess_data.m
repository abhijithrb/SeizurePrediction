
%     This script preprocesses the EEG signals. It first denoises the signals using wavelet transform
%     and then reduces the sampling rate by decimating the signal
%     The script also splits the 10 minute long sequence into 15
%     sequeneces. The new matrix is now a 4D array with size 15, 16, 3200, 1

%%
close all;
clear;
clc;

TRAIN = false; % Indicates whether to preprocess training files or testing files
%% Training directories
if(TRAIN)
        path = {  dir('Data/Raw/Pat1Train/*.mat'),...
                  dir('Data/Raw/Pat2Train/*.mat'),...
                  dir('Data/Raw/Pat3Train/*.mat')...              
                };
else
        path = {   dir('Data/Raw/Pat1Test/*.mat'),...
                  dir('Data/Raw/Pat2Test/*.mat'),...
                  dir('Data/Raw/Pat3Test/*.mat')...
              };
end
%% Directories to save preprocessed files
if(TRAIN)
    save_path = {'Data/Preprocessed2/Pat1Train/', ...
                 'Data/Preprocessed2/Pat2Train/', ...
                 'Data/Preprocessed2/Pat3Train/', ...
                 'Data/Preprocessed2/Testing_Data/', ...
                 'Data/Preprocessed2/Pat2Test/', ...
                 'Data/Preprocessed2/Pat3Test/'...
                 };
else
    save_path = 'Data/Preprocessed/Testing_Data/';
end

data_length = length(path);

% Run the preprocessing functions for each file
for dt = 1 : data_length
    
    subfolder = path{dt};
    len = length(subfolder);
    
    fprintf('Preprocessing folder %d\n', dt)
    
    for idx = 1:len
        sequences = zeros(15, 16, 3200, 1);
        fprintf('Preprocessing file %d of %d\n', idx, len)
        % Read the data ffrom the .mat files
        data_struct = matfile([subfolder(idx).folder, '/', subfolder(idx).name]);
        training_data = data_struct.data';

        fileName = subfolder(idx).name;
        label = fileName(length(fileName) - 4);

        % Perform denoising and decimation
        decimated_data = denoise_decimate(training_data);
        
        % Split the 10 min sequence into 15 time segments
        sq = 1;
        for i = 1 : (size(decimated_data, 2) / 15) : size(decimated_data, 2)
            sequences(sq, :, :, 1) = decimated_data(1 : 16, i : (i-1 + 3200));
            sq = sq + 1;
        end
        
        
        % Save preprocessed file
        fullFileName = sprintf('%s%s', save_path, fileName);

        save(fullFileName, 'sequences', '-v7.3')
    end
end

%%
function decimated_data = denoise_decimate(data)
    % This function denoises a signal using wavelet transform and then performs decimation to reduce
    % sampling rate
    
    n = 5;                  % Decomposition Level - reduce sampling rate by a factor of 5
    w = 'db3';              % Near symmetric wavelet
    [c, l] = wavedec2(data, n, w); % Multilevel 2-D wavelet decomposition.

    opt = 'gbl'; % Global threshold
    [thr, sorh, keepapp] = ddencmp('den','wv',data);
    [denoised_data,~,~,~,~] = wdencmp(opt,c,l,w,n,thr,sorh,keepapp);

    new_samples_size = 80*60*10;

    decimated_data = zeros(16, new_samples_size);

    for channel = 1:16
       decimated_data(channel, :) = decimate( denoised_data(channel, :), 5 ) ;
    end
end