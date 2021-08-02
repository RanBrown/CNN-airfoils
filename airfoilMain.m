%% MAIN DEEPCFD NETWORK FILE

% Set up Saved Folder
directory_path = '/Users/rnbrown/Documents/MATLAB/Airfoil_Project/Saved_results';
timestamp = datetime(datestr(datetime), 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
timestamp.Format = 'MM-dd-yyyy__HH-mm-ss';
[status, msg, msgID] = mkdir(directory_path, string(timestamp));
myFolder = fullfile(directory_path, string(timestamp));
Summary = fopen(fullfile(myFolder, 'summary.txt'),'w');
Network = (fullfile(myFolder, 'trainedNet.mat'));

% Get Data
trainPath = '/Users/rnbrown/Documents/MATLAB/Airfoil_Project/train_data_full.mat';
testPath = '/Users/rnbrown/Documents/MATLAB/Airfoil_Project/test_data_full.mat';
[dsTrain, dsTest, TestY] = airDataLoad(trainPath, testPath);
%% build model

networkName = 'airfoil.m version 1';
%BUILD MODEL
[dlnet] = airfoil();

%% Training
numEpochs = 50;
miniBatchSize = 10;
lossFunctionUsed = 'L1';

% train
[dlnet_trained, learnrate, totalTrainTime] = airTrain(numEpochs, miniBatchSize, dsTrain, dlnet);

save(fullfile(myFolder, 'trainedNet.mat'), 'dlnet_trained')

%% Testing
[YPred, diff] = airTest(miniBatchSize, dsTest, dlnet);
RMSE = sqrt(mean(diff.^2, 'all'));


%% Results
% QUICK CHANGE THE SAMPLE/CHANNEL
sample = 1;
channel = 2;
predAxis = [0, .08];

% SHOW IMAGES
figure
PredImage = image(YPred(:,:,channel,sample), "CDataMapping","scaled");
colorbar
caxis(predAxis) % change color scale

figure
ActImage = image(TestY(:,:,channel,sample), "CDataMapping","scaled");
colorbar
%caxis([-.0,.06])

absErr = abs(YPred(:,:,channel,sample)-TestY(:,:,channel,sample));
figure
ErrImage = image(absErr, "CDataMapping","scaled");
colorbar
caxis([0,.2])

%% Summarize Results

fprintf(Summary, 'SUMMARY OF RESULTS\n');
fprintf(Summary, ['Network used:', ' ', networkName, '\n']);
fprintf(Summary, ['RMSE = ', ' ' , num2str(RMSE),'\n']);
fprintf(Summary, '\n');
fprintf(Summary, 'TRAINING OPTIONS\n');
fprintf(Summary, ['Number of Epochs:', ' ', num2str(numEpochs), '\n']);
fprintf(Summary, ['Size of MiniBatch:', ' ', num2str(miniBatchSize), '\n']);
fprintf(Summary, ['Learning Rate:', ' ', num2str(learnrate), '\n']);
fprintf(Summary, ['Training Time:', ' ', char(string(totalTrainTime, "hh:mm:ss")), '\n']);
fprintf(Summary, ['Loss Function:', ' ', lossFunctionUsed, '\n']);
