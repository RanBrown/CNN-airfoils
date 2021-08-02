function [dsTrain, dsTest, Ytest] = airDataLoad(trainPath, testPath)

% LOAD DATA FILES
train = load(trainPath);
test = load(testPath);
train_set = cell2mat(struct2cell(train));
test_set = cell2mat(struct2cell(test));

%--------------------------------------------------------------------------
% FOR LOOP FOR PREPROCESSING
trainSize = 500;
testStart = trainSize +1;
train_set = train_set(:,:,:,1:trainSize);


% one big array for preprocessing
data = cat(4, train_set, test_set);
Xdata = [];
Ydata = [];
samples = size(data, 4);
loopy = 1:samples;


for i = loopy
% set channels
Xin = data(:,:,1,i);
Yin = data(:,:,2,i);
mask = data(:,:,3,i);
pressure = data(:,:,4,i);
Xout = data(:,:,5,i);
Yout = data(:,:,6,i);
    
% find absolute max for -1 to 1 normalization
	absmaxXin = absMax(Xin);
    absmaxYin = absMax(Yin);
    absmaxMask = absMax(mask);
    absmaxPressure = absMax(pressure);
    absmaxXout = absMax(Xout);
    absmaxYout = absMax(Yout);
    

% remove pressure offset
	pressure = pressure - mean(pressure, [1 2]);

% make dimensionless based on current data set
	v_norm = (absMax(Xin).^2 + absMax(Yin).^2).^.5;
	pressure = pressure./(v_norm.^2);
	Xout = Xout./v_norm;
	Yout = Yout./v_norm;

% normalize -1 to 1 
    Xin = Xin./absmaxXin;
    Yin = Yin./absmaxYin;
    mask = mask./absmaxMask;
    pressure = pressure./absmaxPressure;
    Xout = Xout./absmaxXout;
    Yout = Yout./absmaxYout;
    
    
    XdataSample = cat(3, Xin(:,:,:,:), Yin(:,:,:,:), mask(:,:,:,:));
    YdataSample = cat(3, pressure(:,:,:,:), Xout(:,:,:,:), Yout(:,:,:,:));
    
    
    Xdata = cat(4, Xdata, XdataSample);
    Ydata = cat(4, Ydata, YdataSample);
end

% PREP TRAINING DATA INTO DATASTORES
size(Xdata)
Xtrain = Xdata(:,:,:,1:trainSize);
Ytrain = Ydata(:,:,:,1:trainSize);

dsXTrain = arrayDatastore(Xtrain,'IterationDimension',4);
dsYTrain = arrayDatastore(Ytrain,'IterationDimension',4);

dsTrain = combine(dsXTrain,dsYTrain);


% PREP TESTING DATA INTO DATASTORES
Xtest = Xdata(:,:,:,testStart:end);
Ytest = Ydata(:,:,:,testStart:end);

dsXTest = arrayDatastore(Xtest,'IterationDimension',4);
dsYTest = arrayDatastore(Ytest,'IterationDimension',4);

dsTest = combine(dsXTest,dsYTest);
end


%[TrainInd, TestInd] = dividerand(981, 0.75, 0.25);
%XVal = X(:,:,:,ValInd); MAYBE ADD VALIDATION LATER
%YVal = Y(:,:,:,ValInd);
