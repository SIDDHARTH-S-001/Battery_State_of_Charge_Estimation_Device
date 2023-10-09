url = "https://data.mendeley.com/public-files/datasets/cp3473x7xv/files/ad7ac5c9-2b9e-458a-a91f-6f3da449bdfb/file_downloaded";
downloadFolder = tempdir;
outputFolder = fullfile(downloadFolder, "LGHG2@n10C_to_25degC");

if ~exist(outputFolder, "dir")
    fprintf("Downloading LGHG2@n10C_to_25degC.zip (56 MB) ... ")
    filename = fullfile(downloadFolder, "LGHG2@n10C_to_25degC.zip");
    websave(filename, url);
    unzip(filename, outputFolder);
end

folderTrain = fullfile(outputFolder, "Train");
fdsTrain = fileDatastore(folderTrain, 'ReadFcn', @load);
tdsPredictorsTrain = transform(fdsTrain, @(data) {data.X});
preview(tdsPredictorsTrain);

tdsTargetsTrain = transform(fdsTrain, @(data) {data.Y});
preview(tdsTargetsTrain);

cdsTrain = combine(tdsPredictorsTrain, tdsTargetsTrain);

folderTest = fullfile(outputFolder, "Test");
fdsTest = fileDatastore(folderTest, 'ReadFcn', @load);
tdsPredictorsTest = transform(fdsTest, @(data) {data.X});
preview(tdsPredictorsTest);

tdsTargetsTest = transform(fdsTest, @(data) {data.Y});
preview(tdsTargetsTest);

indices = 1;
vdsPredictors = subset(tdsPredictorsTest, indices);
vdsTargets = subset(tdsTargetsTest, indices);
cdsVal = combine(vdsPredictors, vdsTargets);

numFeatures = 5;
numOutputs = 1;

numHiddenNeurons_1_bounds = [10, 20];
numHiddenNeurons_2_bounds = [15, 30];
numHiddenNeurons_3_bounds = [5, 15];

options = optimoptions('particleswarm', 'SwarmSize', 50, 'MaxIterations', 50, 'UseParallel', true);

bounds = [numHiddenNeurons_1_bounds;
          numHiddenNeurons_2_bounds;
          numHiddenNeurons_3_bounds];

objectiveFcn = @(hyperparams) train_and_evaluate_model(hyperparams, numFeatures, numOutputs, cdsTrain, cdsVal, tdsPredictorsTest, tdsTargetsTest);

numVars = size(bounds, 1);
bestHyperparams = particleswarm(objectiveFcn, numVars, bounds(:, 1), bounds(:, 2), options);

disp('Best Hyperparameters:');
disp(bestHyperparams);

bestModel = train_and_evaluate_model(bestHyperparams, numFeatures, numOutputs, cdsTrain, cdsVal, tdsPredictorsTest, tdsTargetsTest);

for iteration = 1:50
    objectiveFcn = @(hyperparams) train_and_evaluate_model(hyperparams, numFeatures, numOutputs, cdsTrain, cdsVal, tdsPredictorsTest, tdsTargetsTest);

    numVars = size(bounds, 1);
    bestHyperparams = particleswarm(objectiveFcn, numVars, bounds(:, 1), bounds(:, 2), options);

    disp(['Iteration ', num2str(iteration), ' - Best Hyperparameters:']);
    disp(bestHyperparams);

    bestModel = train_and_evaluate_model(bestHyperparams, numFeatures, numOutputs, cdsTrain, cdsVal, tdsPredictorsTest, tdsTargetsTest);

    results{iteration, 1} = bestHyperparams;
    results{iteration, 2} = bestModel;
end

disp('All iterations completed.');

function rmse = train_and_evaluate_model(hyperparams, numFeatures, numOutputs, cdsTrain, cdsVal, tdsPredictorsTest, tdsTargetsTest)
    numHiddenNeurons_1 = round(hyperparams(1));
    numHiddenNeurons_2 = round(hyperparams(2));
    numHiddenNeurons_3 = round(hyperparams(3));

    layers = [
        sequenceInputLayer(numFeatures,Normalization="zscore")
        fullyConnectedLayer(numHiddenNeurons_1)
        tanhLayer                            
        fullyConnectedLayer(numHiddenNeurons_2)
        leakyReluLayer(0.3)
        fullyConnectedLayer(numHiddenNeurons_3)
        leakyReluLayer(0.3)
        fullyConnectedLayer(numOutputs)
        clippedReluLayer(1)
        regressionLayer];

    Epochs = 10;
    miniBatchSize = 1;
    valFrequency = 25;
    InitialLR = 0.01;
    LRDropPeriod = 25;
    LRDropFactor = 0.99;

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
        Verbose=0, ...
        Shuffle="every-epoch",...
        Plots="training-progress", ...
        ExecutionEnvironment="cpu");

    net = trainNetwork(cdsTrain, layers, options);
    % Rest of the function remains the same...
end
