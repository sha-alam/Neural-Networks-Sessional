% Load the Iris dataset
load fisheriris
X = meas;  % Features
y = species;  % Target variable

% Standardize the features
X_scaled = (X - mean(X)) ./ std(X);

disp("Before:");
disp(size(X_scaled));

% Perform PCA
[coeff, X_pca, latent, ~, explained] = pca(X_scaled);  % Reduce to 2 dimensions

% Select the first two principal components
X_pca = X_pca(:, 1:2);

disp("After:");
disp(size(X_pca));

% Create a table for the reduced data
pca_table = array2table(X_pca, 'VariableNames', {'PrincipalComponent1', 'PrincipalComponent2'});
pca_table.Target = categorical(y);

% Plot the PCA results
figure;
gscatter(pca_table.PrincipalComponent1, pca_table.PrincipalComponent2, pca_table.Target, [], 'osd', [], 'off');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('PCA of Iris Dataset');
legend('Location', 'best');
grid on;

% Explained variance
explained_variance = explained(1:2) / sum(explained);
total_explained_variance = sum(explained_variance);
disp(['Explained variance by component: ', num2str(explained_variance')]);
disp(['Total explained variance: ', num2str(total_explained_variance)]);
