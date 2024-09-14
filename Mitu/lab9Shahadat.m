% Load the dataset
data = readtable('user-data.csv');

% Handle categorical variables using label encoding
varNames = data.Properties.VariableNames;
for i = 1:width(data)
    if iscellstr(data.(i))
        data.(i) = grp2idx(data.(i));  % Convert categorical variables to numeric
    end
end

% Display the first few rows of data
disp(head(data));

% Extract features and target variable
X = table2array(data(:, ~ismember(varNames, {'user_id', 'purchased'})));  % Features
y = table2array(data(:, 'purchased'));  % Target variable

% Split the dataset into training and test sets (80% training, 20% testing)
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
X_test = X(test(cv), :);
y_train = y(training(cv), :);
y_test = y(test(cv), :);

% Feature scaling
X_train = (X_train - mean(X_train)) ./ std(X_train);
X_test = (X_test - mean(X_test)) ./ std(X_test);

% Train the SVM classifier
SVMModel = fitcsvm(X_train, y_train, 'KernelFunction', 'linear', 'Standardize', true, 'ClassNames', [0, 1]);

% Predict the test set results
y_pred = predict(SVMModel, X_test);

% Evaluate the model using confusion matrix and accuracy
cm = confusionmat(y_test, y_pred);
accuracy = sum(diag(cm)) / sum(cm(:));

% Print the confusion matrix and accuracy
disp('Confusion Matrix:');
disp(cm);
disp(['Accuracy: ', num2str(accuracy)]);

% Visualize the confusion matrix
figure;
heatmap({'Not Purchased', 'Purchased'}, {'Not Purchased', 'Purchased'}, cm, 'Colormap', parula, 'ColorbarVisible', 'on');
xlabel('Predicted');
ylabel('Actual');
title('Confusion Matrix');
