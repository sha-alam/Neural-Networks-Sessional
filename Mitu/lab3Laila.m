function lab3Laila()
    % Input and target datasets
    X = [0 0 1;
         0 1 1;
         1 0 1;
         1 1 1];

    D = [0; 0; 1; 1];  % Target output

    % Initialize weights (random values between -1 and 1)
    W = 2 * rand(1, 3) - 1;

    % Training loop
    for epoch = 1:10000
        W = SGD(W, X, D);  % Call SGD function
    end

    % Prediction (testing the network)
    YY = [];
    N = 4;
    for k = 1:N
        x = X(k, :)';  % Input vector
        v = W * x;  % Forward pass
        y = sigmoid(v);  % Sigmoid activation
        YY = [YY y];  % Store the output
    end

    disp('Prediction of this net:');
    disp(YY > 0.9);  % Display predictions (threshold at 0.9)
end

function [W] = SGD(W, X, D)
    alpha = 0.9;  % Learning rate
    N = 4;  % Number of samples
    for k = 1:N
        x = X(k, :)';  % Take input vector
        d = D(k);  % Target output
        
        % Forward pass (dot product)
        v = W * x;
        y = sigmoid(v);  % Sigmoid activation
        
        % Compute error
        e = d - y;
        
        % Compute delta (for weight adjustment)
        delta = y * (1 - y) * e;
        
        % Compute weight updates
        dW = alpha * delta * x;
        
        % Update weights using vectorized operation
        W = W + dW';
    end
end

% Sigmoid activation function
function y = sigmoid(v)
    y = 1 / (1 + exp(-v));
end
