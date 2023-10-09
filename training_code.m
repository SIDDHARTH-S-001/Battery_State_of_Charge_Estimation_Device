url = "https://data.mendeley.com/public-files/datasets/cp3473x7xv/files/ad7ac5c9-2b9e-458a-a91f-6f3da449bdfb/file_downloaded";
downloadFolder = tempdir;
outputFolder = fullfile(downloadFolder, "LGHG2@n10C_to_25degC");
if ~exist(outputFolder,"dir")
    fprintf("Downloading LGHG2@n10C_to_25degC.zip (56 MB) ... ")
    filename = fullfile(downloadFolder,"LGHG2@n10C_to_25degC.zip");
    websave(filename,url);
    unzip(filename,outputFolder)
end

folderTrain = fullfile(outputFolder,"Train");
fdsTrain = fileDatastore(folderTrain, ReadFcn=@load); 
tdsPredictorsTrain = transform(fdsTrain, @(data) {data.X});
preview(tdsPredictorsTrain)
tdsTargetsTrain = transform(fdsTrain, @(data) {data.Y});
preview(tdsTargetsTrain)
cdsTrain = combine(tdsPredictorsTrain,tdsTargetsTrain);

folderTest = fullfile(outputFolder,"Test");
fdsTest = fileDatastore(folderTest, ReadFcn=@load);
tdsPredictorsTest = transform(fdsTest, @(data) {data.X});
preview(tdsPredictorsTest) 
tdsTargetsTest = transform(fdsTest,@(data) {data.Y});
preview(tdsTargetsTest)

indices = 1;
vdsPredictors = subset(tdsPredictorsTest,indices);
vdsTargets = subset(tdsTargetsTest,indices);
cdsVal = combine(vdsPredictors,vdsTargets);

numFeatures = 5;
numOutputs = 1;
numHiddenNeurons_1 = 49;
numHiddenNeurons_2 = 34;
numHiddenNeurons_3 = 15; 
layers = [
    sequenceInputLayer(numFeatures,Normalization="zscore") % zerocenter -> x - mean, zscore = (x - mean)/std-dev
    fullyConnectedLayer(numHiddenNeurons_1)
    tanhLayer                            
    fullyConnectedLayer(numHiddenNeurons_2)
    leakyReluLayer(0.3) % A leaky ReLU layer performs a threshold operation, where any input value less than zero is multiplied by a fixed scalar.
    fullyConnectedLayer(numHiddenNeurons_3)
    leakyReluLayer(0.3)
    fullyConnectedLayer(numOutputs)
    clippedReluLayer(1) % A clipped ReLU layer performs a threshold operation, where any input value less than zero is set to zero and any value above the clipping ceiling is set to that clipping ceiling.
    regressionLayer]; % regression layer computes half-mean squared loss.

Epochs = 150;
miniBatchSize = 1; % 0.01, 0.01*0.5 = 0.005, 0.01*0.5*0.5 = 0.025,0.00125, 0.000625 
LRDropPeriod = 250; 
InitialLR = 0.01;
LRDropFactor = 0.9826; 
valFrequency = 25; 

options = trainingOptions("adam", ...                 
    MaxEpochs=Epochs, ...
    GradientThreshold=1, ...
    InitialLearnRate=InitialLR, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=LRDropPeriod, ...
    LearnRateDropFactor=LRDropFactor, ...
    ValidationData=cdsVal, ...
    ValidationFrequency=valFrequency, ...
    MiniBatchSize=miniBatchSize, ...
    L2Regularization=0.0001, ...
    Verbose=true, ... 
    Plots="training-progress");

net = trainNetwork(cdsTrain,layers,options);
YPred = predict(net,tdsPredictorsTest,MiniBatchSize=1);
YTarget = readall(tdsTargetsTest);

figure

nexttile
plot(YPred{1})
hold on
plot(YTarget{1})
legend(["Predicted" "Target"], Location="Best")
ylabel("SOC")
xlabel("Time(s)")
title("n10degC")

nexttile
plot(YPred{2})
hold on
plot(YTarget{2})
legend(["Predicted" "Target"], Location="Best")
ylabel("SOC")
xlabel("Time(s)")
title("0degC")

nexttile
plot(YPred{3})
hold on
plot(YTarget{3})
legend(["Predicted" "Target"], Location="Best")
ylabel("SOC")
xlabel("Time(s)")
title("10degC")

nexttile
plot(YPred{4})
hold on
plot(YTarget{4})
legend(["Predicted" "Target"], Location="Best")
ylabel("SOC")
xlabel("Time(s)")
title("25degC")

Err_n10degC = YPred{1} - YTarget{1};
Err_0degC = YPred{2} - YTarget{2};
Err_10degC = YPred{3} - YTarget{3};
Err_25degC = YPred{4} - YTarget{4};
RMSE_n10degC = sqrt(mean(Err_n10degC.^2))*100;
RMSE_0degC = sqrt(mean(Err_0degC.^2))*100;
RMSE_10degC = sqrt(mean(Err_10degC.^2))*100;
RMSE_25degC = sqrt(mean(Err_25degC.^2))*100;
MAX_n10degC = max(abs(Err_n10degC))*100;
MAX_0degC = max(abs(Err_0degC))*100;
MAX_10degC = max(abs(Err_10degC))*100;
MAX_25degC = max(abs(Err_25degC))*100;

temp = [-10,0,10,25];
figure
nexttile
bar(temp,[RMSE_n10degC,RMSE_0degC,RMSE_10degC,RMSE_25degC])
ylabel("RMSE (%)")
xlabel("Temperature (C)")

nexttile
bar(temp,[MAX_n10degC,MAX_0degC,MAX_10degC,MAX_25degC])
ylabel("MAX (%)")
xlabel("Temperature (C)")