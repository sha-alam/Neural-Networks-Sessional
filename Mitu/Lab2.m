% XOR implementation using McCulloch-Pitts neuron in MATLAB

% Sigmoid activation function and its derivative (for training)
sigmoid = @(x) 1 ./ (1 + exp(-x));
sigmoid_derivative = @(x) x .* (1 - x);

% XOR function dataset
inputs = [0 0; 0 1; 1 0; 1 1];
targets = [0; 1; 1; 0];

% Neural network parameters
input_layer_size = 2;
hidden_layer_size = 2;
output_layer_size = 1;
learning_rate = 0.1;
max_epochs = 10000;

% Initialize weights and biases with random values
rng(42);  % Seed for reproducibility
weights_input_hidden = randn(input_layer_size, hidden_layer_size);
bias_hidden = randn(1, hidden_layer_size);

weights_hidden_output = randn(hidden_layer_size, output_layer_size);
bias_output = randn(1, output_layer_size);

convergence_curve = [];

% Training the neural network
for epoch = 1:max_epochs
    misclassified = 0;
    for i = 1:size(inputs, 1)
        % Forward pass
        hidden_layer_input = inputs(i, :) * weights_input_hidden + bias_hidden;
        hidden_layer_output = sigmoid(hidden_layer_input);

        output_layer_input = hidden_layer_output * weights_hidden_output + bias_output;
        predicted_output = sigmoid(output_layer_input);

        % Backpropagation
        error = targets(i) - predicted_output;

        if targets(i) ~= round(predicted_output)
            misclassified = misclassified + 1;
        end

        output_delta = error * sigmoid_derivative(predicted_output);
        hidden_delta = (output_delta * weights_hidden_output') .* sigmoid_derivative(hidden_layer_output);

        % Update weights and biases
        weights_hidden_output = weights_hidden_output + hidden_layer_output' * output_delta * learning_rate;
        bias_output = bias_output + output_delta * learning_rate;

        weights_input_hidden = weights_input_hidden + inputs(i, :)' * hidden_delta * learning_rate;
        bias_hidden = bias_hidden + hidden_delta * learning_rate;
    end
    
    accuracy = (size(inputs, 1) - misclassified) / size(inputs, 1);
    convergence_curve = [convergence_curve; accuracy];

    if misclassified == 0
        fprintf('Converged in %d epochs.\n', epoch);
        break;
    end
end

% Decision boundary line
x = linspace(-0.5, 1.5, 100);
y1 = (-weights_input_hidden(1, 1) * x - bias_hidden(1)) / weights_input_hidden(2, 1);
y2 = (-weights_input_hidden(1, 2) * x - bias_hidden(2)) / weights_input_hidden(2, 2);

% Plot convergence curve
figure;
plot(1:length(convergence_curve), convergence_curve, 'LineWidth', 1.5);
xlabel('Epoch');
ylabel('Accuracy');
title('Convergence Curve');
grid on;

% Plot the decision boundary line and data points
figure;
plot(x, y1, 'r', 'DisplayName', 'Decision Boundary 1', 'LineWidth', 1.5); hold on;
plot(x, y2, 'b', 'DisplayName', 'Decision Boundary 2', 'LineWidth', 1.5);
scatter(inputs(targets == 1, 1), inputs(targets == 1, 2), 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'cyan', 'DisplayName', 'Target 1 (1)');
scatter(inputs(targets == 0, 1), inputs(targets == 0, 2), 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'black', 'DisplayName', 'Target 0 (0)');
xlabel('Input 1');
ylabel('Input 2');
title('XOR Function Decision Boundary');
legend;
grid on;
hold off;
