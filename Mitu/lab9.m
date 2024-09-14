% Number of samples to generate
num_samples = 1000;

% Generate synthetic data
Age = randi([30, 80], num_samples, 1);  % Age between 30 and 80
BloodPressure = randi([90, 200], num_samples, 1);  % Systolic BP between 90 and 200 mmHg
Cholesterol = randi([150, 300], num_samples, 1);  % Cholesterol between 150 and 300 mg/dL
RestingHeartRate = randi([50, 100], num_samples, 1);  % Resting heart rate between 50 and 100 bpm
Smoking = randi([0, 1], num_samples, 1);  % 1 for smoker, 0 for non-smoker
FamilyHistory = randi([0, 1], num_samples, 1);  % 1 if family history of heart disease, 0 otherwise
ExerciseLevel = randi([0, 2], num_samples, 1);  % 0 = Low, 1 = Moderate, 2 = High exercise level

% Simulate the presence of heart disease
% Logic: People with high BP, cholesterol, age, smokers, or low exercise more likely to have heart disease
HeartDisease = (BloodPressure > 140 | Cholesterol > 240 | Age > 55 | Smoking == 1 | FamilyHistory == 1) & ExerciseLevel == 0;

% Create a table
data = table(Age, BloodPressure, Cholesterol, RestingHeartRate, Smoking, FamilyHistory, ExerciseLevel, HeartDisease);

% Display first few rows
disp(data(1:10, :));

% Save the dataset to a CSV file
writetable(data, 'D:\Study_Object\4_2_Course\NeuralNetworks\LbMitu.csv');

% Load the heart disease dataset
data = readtable('D:\Study_Object\4_2_Course\NeuralNetworks\LbMitu.csv');

% Display the first few rows of the dataset
disp(data(1:5, :));

% Extract features (X) and labels (Y)
% Features: Age, BloodPressure, Cholesterol, RestingHeartRate, Smoking, FamilyHistory, ExerciseLevel
X = data{:, 1:end-1};  % All columns except 'HeartDisease'
Y = data.HeartDisease;  % 'HeartDisease' column

% Split the data into training and testing sets
cv = cvpartition(size(X, 1), 'HoldOut', 0.3);  % 70% training, 30% testing
XTrain = X(training(cv), :);
YTrain = Y(training(cv), :);
XTest = X(test(cv), :);
YTest = Y(test(cv), :);

% Train an SVM model
svmModel = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear', 'Standardize', true);

% Perform cross-validation
CVSVMModel = crossval(svmModel);
loss = kfoldLoss(CVSVMModel);
fprintf('Cross-validation loss: %.4f\n', loss);

% Make predictions on the test set
YPred = predict(svmModel, XTest);

% Calculate accuracy
accuracy = sum(YPred == YTest) / length(YTest) * 100;
fprintf('Test set accuracy: %.2f%%\n', accuracy);

% Display confusion matrix
confMat = confusionmat(YTest, YPred);
disp('Confusion Matrix:');
disp(confMat);

% Plot confusion matrix
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix for Heart Disease Prediction');

