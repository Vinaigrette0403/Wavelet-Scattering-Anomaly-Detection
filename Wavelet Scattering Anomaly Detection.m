% 20195095 박상현
% 웨이블릿 스캐터링 시퀀스와 컨볼루션 오토인코더를 이용한 이상 탐지
% 2025.12.05

url = 'https://www.mathworks.com/supportfiles/audio/AirCompressorDataset/AirCompressorDataset.zip';
downloadFolder = fullfile(tempdir,'AirCompressorDataSet');
if ~exist(fullfile(tempdir,'AirCompressorDataSet'),'dir') 
    loc = websave(downloadFolder,url); 
    unzip(loc,fullfile(tempdir,'AirCompressorDataSet')) 
end

rng("default")
ads = audioDatastore(downloadFolder,IncludeSubfolders=true,...
    LabelSource="foldernames"); 
    
[adsTrain,adsValidation,adsTest] = splitEachLabel(ads,0.5,0.2,0.3,...
    Include=["Healthy","Bearing"]);

C = categories(adsTrain.Labels);
categoriesToRemove = C(~ismember(C,unique(adsTrain.Labels)));
uniqueLabels = unique(removecats(adsTrain.Labels,categoriesToRemove));
tblTrain = countEachLabel(adsTrain);
tblValidation = countEachLabel(adsValidation);
tblTest = countEachLabel(adsTest);
H = bar(uniqueLabels,[tblTrain.Count, tblValidation.Count, tblTest.Count],'stacked');
ax = gca;
ax.YTick = [113 113+45 113+45+67];
ax.YLabel.String = "Cumulative Number of Examples";
legend(H,["Training","Validation","Test"],Location="NorthEastOutside",FontSize=12)

tiledlayout(2,2)
for n = 1:numel(uniqueLabels)
    idx = find(adsTrain.Labels==uniqueLabels(n),1);
    [x,fs] = audioread(adsTrain.Files{idx});    
    t = (0:size(x,1)-1)/fs;
    nexttile
    plot(t,x);
    grid on
    xlabel("Time (seconds)")
    ylabel("Amplitude")
    title(string(uniqueLabels(n)));
    nexttile
    pspectrum(x,fs)
    title(string(uniqueLabels(n)));
end

N = 5e4;
sn = waveletScattering(SignalLength=N,SamplingFrequency=fs,...
    InvarianceScale=0.3,OversamplingFactor=1); 
numCoefficients(sn)

[~,npaths] = paths(sn);
sum(npaths)

useGPU = false; 
batchsize = 64;
scTrain = [];
reset(adsTrain)
while hasdata(adsTrain)
    sc = helperbatchscatfeatures(adsTrain,sn,N,batchsize,useGPU);
    scTrain = cat(3,scTrain,sc);
end

scValidation = [];
reset(adsValidation)
while hasdata(adsValidation)
   sc = helperbatchscatfeatures(adsValidation,sn,N,batchsize,useGPU);
   scValidation = cat(3,scValidation,sc); 
end
scTest = [];
reset(adsTest)
while hasdata(adsTest)
   sc = helperbatchscatfeatures(adsTest,sn,N,batchsize,useGPU);
   scTest = cat(3,scTest,sc); 
end

trainFeatures = scTrain(2:end,:,:);
trainFeatures = squeeze(num2cell(trainFeatures,[1 2])); 
trainLabels = adsTrain.Labels;
validationFeatures = scValidation(2:end,:,:);
validationFeatures = squeeze(num2cell(validationFeatures,[1 2]));
validationLabels = adsValidation.Labels;
testFeatures = scTest(2:end,:,:);
testFeatures = squeeze(num2cell(testFeatures,[1 2]));
testLabels = adsTest.Labels;

trainNormal = trainFeatures(trainLabels=="Healthy");
trainFaulty = trainFeatures(trainLabels=="Bearing");
validationNormal = validationFeatures(validationLabels=="Healthy");
validationFaulty = validationFeatures(validationLabels=="Bearing");
testNormal = testFeatures(testLabels=="Healthy");
testFaulty = testFeatures(testLabels=="Bearing");
testLabels = removecats(testLabels,categoriesToRemove);

trainNormal = cellfun(@transpose,trainNormal,UniformOutput=false);
validationNormal = cellfun(@transpose,validationNormal,UniformOutput=false);
testNormal = cellfun(@transpose,testNormal,UniformOutput=false);
trainFaulty = cellfun(@transpose,trainFaulty,UniformOutput=false);
validationFaulty = cellfun(@transpose,validationFaulty,UniformOutput=false);
testFaulty = cellfun(@transpose,testFaulty,UniformOutput=false);

testFaulty = cat(1,trainFaulty,validationFaulty,testFaulty);

reset(adsTrain)
reset(adsValidation)
reset(adsTest)
trainSequences = readall(adsTrain);
validationSequences = readall(adsValidation);
testSequences = readall(adsTest);

trainSequences = cellfun(@(x)(x-mean(x))./std(x),trainSequences,UniformOutput=false);
validationSequences = cellfun(@(x)(x-mean(x))./std(x),validationSequences,UniformOutput=false);
testSequences = cellfun(@(x)(x-mean(x))./std(x),testSequences,UniformOutput=false);
normalTrainSequences = trainSequences(adsTrain.Labels == "Healthy");
normalValidationSequences = validationSequences(adsValidation.Labels=="Healthy");
normalTestSequences = testSequences(adsTest.Labels=="Healthy");
faultyTrainSequences = trainSequences(adsTrain.Labels == "Bearing");
faultyValidationSequences = trainSequences(adsValidation.Labels == "Bearing");
faultyTestSequences = testSequences(adsTest.Labels=="Bearing");

faultySequences = cat(1,faultyTrainSequences,faultyValidationSequences,faultyTestSequences);

trainModels = true;
if trainModels
    numChannels = size(trainNormal{1},2);
    dsadSCAT = deepSignalAnomalyDetector(numChannels,WindowLength="fullSignal");
end

if trainModels
    numChannels = 1;
    dsadRAW = deepSignalAnomalyDetector(numChannels,WindowLength="fullSignal");
end

if trainModels
    optsSCAT = trainingOptions("adam", MaxEpochs=50,MiniBatchSize=16, ...
        ValidationData={validationNormal,validationNormal},...
        Shuffle="every-epoch",...
        OutputNetwork = "best-validation-loss",...
        Verbose=false,Plots="training-progress"); 
    optsRAW = trainingOptions("adam",MaxEpochs=50,MiniBatchSize=16, ...
        ValidationData={normalValidationSequences,normalValidationSequences},... 
        OutputNetwork = "best-validation-loss",...
        Verbose=false,Plots="training-progress");
end

if trainModels
    trainDetector(dsadSCAT,trainNormal,optsSCAT)
end

if trainModels
    trainDetector(dsadRAW,normalTrainSequences,optsRAW)
end

if ~trainModels
    load dsadSCAT.mat %#ok<*UNRCH>
    load dsadRAW.mat
end

figure
figh = plotLossDistribution(dsadSCAT,testNormal,testFaulty);
figh.Children(1).String = ["Healthy","Bearing","Normal CDF","Faulty CDF"];
ax = gca;
ax.Title.String = "Reconstruction Loss -- Scattering Sequences";

fullTestLabels = cat(1,trainLabels(trainLabels=="Bearing"),validationLabels(validationLabels=="Bearing"),testLabels(testLabels=="Bearing"),testLabels(testLabels=="Healthy"));
fullTestLabels = removecats(fullTestLabels,categoriesToRemove);
predNormalSCAT = detect(dsadSCAT,testNormal);
predBearingSCAT = detect(dsadSCAT,testFaulty);
predSCAT = cat(1,cell2mat(predBearingSCAT),cell2mat(predNormalSCAT));
predSCAT = categorical(predSCAT,[1 0],["Bearing" "Healthy"]);
cm = confusionchart(fullTestLabels,predSCAT);
cm.RowSummary = "row-normalized";
cm.ColumnSummary = "column-normalized";
cm.Title = "Wavelet Scattering Sequences with deepSignalAnomalyDetector"