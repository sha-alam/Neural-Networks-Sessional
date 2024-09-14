% MATLAB code for Perceptron to compute AND function with bipolar inputs and targets
clc;
clear;

% Bipolar inputs and corresponding targets for AND function
inputs = [-1, -1; -1, +1; +1, -1; +1, +1];  % Inputs
targets = [-1; -1; -1; +1];                 % Targets

% Initialize weights and bias
weights = [0, 0];                            % Weight vector [W1, W2]
bias = 0;                                    % Bias
eta = 1;                                     % Learning rate
epochs = 10;                                 % Number of iterations to train
convergence_curve = zeros(epochs, 1);        % Track total errors per epoch

% Perceptron training algorithm
for epoch = 1:epochs
    total_error = 0;
    
    for i = 1:size(inputs, 1)
        % Extract input sample
        x1 = inputs(i, 1);
        x2 = inputs(i, 2);
        
        % Compute the output using the sign function
        output = sign(weights(1) * x1 + weights(2) * x2 + bias);
        
        % In case output is 0, treat it as +1 to match bipolar targets
        if output == 0
            output = 1;
        end
        
        % Calculate error
        error = targets(i) - output;
        total_error = total_error + abs(error);
        
        % Update weights and bias if there's an error
        if error ~= 0
            weights(1) = weights(1) + eta * error * x1;
            weights(2) = weights(2) + eta * error * x2;
            bias = bias + eta * error;
        end
    end
    
    % Record total error for convergence curve
    convergence_curve(epoch) = total_error;
end

% Plot convergence curve
figure;
plot(1:epochs, convergence_curve, '-o', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Total Error');
title('Convergence Curve');
grid on;

% Plot decision boundary
figure;
hold on;

% Scatter plot of input points
scatter(inputs(targets==-1, 1), inputs(targets==-1, 2), 'o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b', 'DisplayName', 'Target -1');
scatter(inputs(targets==1, 1), inputs(targets==1, 2), 'o', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', 'DisplayName', 'Target +1');

% Plot decision boundary line
x_line = -1.5:0.1:1.5;  % X-axis values
y_line = (-bias - weights(1)*x_line) / weights(2);  % Y-axis values based on the decision boundary equation
plot(x_line, y_line, 'g', 'LineWidth', 2, 'DisplayName', 'Decision Boundary');  % Plot decision boundary

xlabel('Input 1');
ylabel('Input 2');
title('Decision Boundary Line');
legend('show');  % Show the legend with all labels
axis([-1.5 1.5 -1.5 1.5]);
grid on;
hold off;




