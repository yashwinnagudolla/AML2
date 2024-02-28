% Load and split data into X1, X2, y1, y2

% Part (a) - Feature normalization
X1_norm = 1 + 99*(X1 - min(X1))/range(X1); 
X2_norm = 1 + 99*(X2 - min(X2))/range(X2);

X_norm = [X1_norm; X2_norm]; 

K = X_norm*X_norm';
% Compute alphas, b, predictions etc. 

% Part (b) - Distance normalization
D = sqrt(diag(K))';  
D_norm = 1 + 99*(D - min(D))/range(D);

K_norm = bsxfun(@times, K, 1./D_norm');
K_norm = bsxfun(@times, K_norm, 1./D_norm);
