% Inputs and target values
x1 = 0.05; 
x2 = 0.10;
inputs = [x1; x2];  % Input vector

T1 = 0.01; 
T2 = 0.99;
targets = [T1; T2];  % Target vector

% Initial weights and biases
w1 = 0.15; w2 = 0.20;  % Weights for input to H1
w3 = 0.25; w4 = 0.30;  % Weights for input to H2
w5 = 0.40; w6 = 0.45;  % Weights from H1, H2 to y1
w7 = 0.50; w8 = 0.55;  % Weights from H1, H2 to y2
b1 = 0.35;  % Bias for hidden layer H1, H2
b2 = 0.60;  % Bias for output layer y1, y2

% Set hyperparameters
learning_rate = 0.5;
epochs = 10000;  % Number of iterations

% Training loop
for epoch = 1:epochs
    % Forward pass
    h1_input = w1 * x1 + w2 * x2 + b1;
    h2_input = w3 * x1 + w4 * x2 + b1;
    
    h1_output = sigmoid(h1_input);
    h2_output = sigmoid(h2_input);
    
    y1_input = w5 * h1_output + w6 * h2_output + b2;
    y2_input = w7 * h1_output + w8 * h2_output + b2;
    
    y1_output = sigmoid(y1_input);
    y2_output = sigmoid(y2_input);
    
    % Calculate the error (Mean Squared Error)
    error1 = (y1_output - T1)^2;
    error2 = (y2_output - T2)^2;
    total_error = (error1 + error2) / 2;
    
    % Backward pass (Backpropagation)
    % Calculate gradients for output layer
    delta_y1 = (y1_output - T1) * sigmoid_derivative(y1_output);
    delta_y2 = (y2_output - T2) * sigmoid_derivative(y2_output);
    
    % Calculate gradients for hidden layer
    delta_h1 = (delta_y1 * w5 + delta_y2 * w7) * sigmoid_derivative(h1_output);
    delta_h2 = (delta_y1 * w6 + delta_y2 * w8) * sigmoid_derivative(h2_output);
    
    % Update weights and biases
    w1 = w1 - learning_rate * delta_h1 * x1;
    w2 = w2 - learning_rate * delta_h1 * x2;
    w3 = w3 - learning_rate * delta_h2 * x1;
    w4 = w4 - learning_rate * delta_h2 * x2;
    
    w5 = w5 - learning_rate * delta_y1 * h1_output;
    w6 = w6 - learning_rate * delta_y1 * h2_output;
    w7 = w7 - learning_rate * delta_y2 * h1_output;
    w8 = w8 - learning_rate * delta_y2 * h2_output;
    
    b1 = b1 - learning_rate * (delta_h1 + delta_h2);
    b2 = b2 - learning_rate * (delta_y1 + delta_y2);
    
    % Display the error every 1000 epochs
    if mod(epoch, 1000) == 0
        fprintf('Epoch %d, Total Error: %f\n', epoch, total_error);
    end
end

% Final outputs
fprintf('Final output y1: %f\n', y1_output);
fprintf('Final output y2: %f\n', y2_output);

% Sigmoid function definition
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

% Derivative of sigmoid function
function y = sigmoid_derivative(x)
    y = x .* (1 - x);
end
