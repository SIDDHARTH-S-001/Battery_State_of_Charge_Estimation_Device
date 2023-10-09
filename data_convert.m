url = "https://data.mendeley.com/public-files/datasets/cp3473x7xv/files/ad7ac5c9-2b9e-458a-a91f-6f3da449bdfb/file_downloaded";
downloadFolder = tempdir;
outputFolder = fullfile(downloadFolder, "LGHG2@n10C_to_25degC");
if ~exist(outputFolder, "dir")
    fprintf("Downloading LGHG2@n10C_to_25degC.zip (56 MB) ... ")
    filename = fullfile(downloadFolder, "LGHG2@n10C_to_25degC.zip");
    websave(filename, url);
    unzip(filename, outputFolder);
end
%%
folderTrain = fullfile(outputFolder, "Train");
fdsTrain = fileDatastore(folderTrain, ReadFcn=@load);
tdsPredictorsTrain = transform(fdsTrain, @(data) {data.X});
tdsTargetsTrain = transform(fdsTrain, @(data) {data.Y});
cdsTrain = combine(tdsPredictorsTrain, tdsTargetsTrain);
%%
folderTest = fullfile(outputFolder, "Test");
fdsTest = fileDatastore(folderTest, ReadFcn=@load);
tdsPredictorsTest = transform(fdsTest, @(data) {data.X});
tdsTargetsTest = transform(fdsTest, @(data) {data.Y});
%%
indices = 1;
vdsPredictors = subset(tdsPredictorsTest, indices);
vdsTargets = subset(tdsTargetsTest, indices);
cdsVal = combine(vdsPredictors, vdsTargets);
%%
%train_data = load('LGH2@LGHG2@n10C_to_25degC/Train/TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat')
train_data = load('TRAIN_LGHG2@n10degC_to_25degC_Norm_5Inputs.mat')
writetable(train_data, 'train_data.csv')

%% Assuming you have a table named 'train_data'
% Assuming you have a table named 'train_data'
filename = 'train_data.csv'; % Specify the desired output CSV filename

% Convert struct variable to a compatible format
struct_field_name = 'NameOfStructField'; % Replace with the actual field name
train_data_with_cell = train_data;
train_data_with_cell.(struct_field_name) = cellstr(train_data.(struct_field_name));

% Use writestruct to write the table with struct variables to a CSV file
writestruct(train_data_with_cell, filename);

disp(['Table data has been written to ' filename]);





