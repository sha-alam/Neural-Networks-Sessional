% Sigmoid activation function and its derivative (for training)
sigmoid = @(x) 1 ./ (1 + exp(-x));
sigmoid_derivative = @(x) x .* (1 - x);

% Input and target datasets
X_input = [0 0 1; 0 1 1; 1 0 1; 1 1 1];
D_target = [0; 0; 1; 1];

% Neural network parameters
input_layer_size = 3;
output_layer_size = 1;
learning_rate = 0.1;
max_epochs = 10000;

% Initialize weights with random values
rng(42); % For reproducibility
weights = randn(input_layer_size, output_layer_size);

% Training the neural network with SGD
for epoch = 1:max_epochs
    error_sum = 0;
    
    for i = 1:size(X_input, 1)
        % Forward pass
        input_data = X_input(i, :);
        target_data = D_target(i, :);

        net_input = input_data * weights;
        predicted_output = sigmoid(net_input);

        % Calculate error
        error = target_data - predicted_output;
        error_sum = error_sum + abs(error);

        % Update weights using the delta learning rule
        weight_update = learning_rate * error * input_data';
        weights = weights + weight_update;

    end

    % Check for convergence
    if error_sum < 0.01
        fprintf('Converged in %d epochs.\n', epoch);
        break;
    end
end

% Test data
test_data = X_input;

% Use the trained model to recognize target function
disp('Target Function Test:');
for i = 1:size(test_data, 1)
    input_data = test_data(i, :);
    net_input = input_data * weights;
    predicted_output = sigmoid(net_input);

    fprintf('Input: [%d %d %d] -> Output: %d\n', input_data, round(predicted_output));
end
