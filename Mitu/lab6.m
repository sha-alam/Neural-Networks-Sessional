% Set paths to your dataset
fruitsPath = 'D:\Study_Object\4_2_Course\NeuralNetworks\Neural network lab\path_to_sample_fruit_image\path_to_sample_fruit_image';

 

% Create imageDatastore for the training dataset
imdsTrain = imageDatastore(fruitsPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

 
numClasses = 3;
% Define CNN architecture
layers = [
    imageInputLayer([224 224 3])

    convolution2dLayer(3, 32, 'Padding', 'same')
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 128, 'Padding', 'same')
    reluLayer

    fullyConnectedLayer(64)
    reluLayer

    fullyConnectedLayer(numClasses)  % numClasses is the number of fruit classes
    softmaxLayer
    classificationLayer
];

 

% Set training options
miniBatchSize = 32;
numEpochs = 20;
initialLearnRate = 1e-4;

 

options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', numEpochs, ...
    'InitialLearnRate', initialLearnRate, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsTrain, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

 

% Train the CNN
net = trainNetwork(imdsTrain, layers, options);

 

% Save the trained network
save('D:\Study_Object\4_2_Course\NeuralNetworks\Neural network lab\path_to_sample_fruit_image\path_to_sample_fruit_image', 'net');
% Load and preprocess a random sample fruit image for classification
sampleImageFile = imdsTrain.Files{randi(length(imdsTrain.Files))};
sampleImage = imread(sampleImageFile);

 

inputImage = imresize(sampleImage, [224 224]);

 

% Classify the sample image
predictedLabel = classify(net, inputImage);

 

% Display the predicted label and the sample image
disp(['Predicted label: ' char(predictedLabel)]);
imshow(sampleImage);
title(['Predicted: ' char(predictedLabel)]);

