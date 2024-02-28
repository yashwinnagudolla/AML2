% Load data
data = load('./iris.csv');

% Extract features and labels
X = data(:,1:4);
y = data(:,5);

% Number of examples
n1 = 50; 
n2 = 50;
n3 = 50;

% Class indexes 
I1 = 1:n1;
I2 = n1+1:n1+n2;
I3 = n1+n2+1:n1+n2+n3;

% Setosa vs others 
alpha1 = zeros(150,1);
alpha1(I1) = 1/n1;
alpha2 = zeros(150,1); 
alpha2(I2) = 1/n2;
alpha2(I3) = 1/n3;

K = X*X';  
b = (alpha1'*K*alpha1 - alpha2'*K*alpha2)/2;

f = @(x) sign(sum(bsxfun(@times,alpha1-alpha2,K(:,x)),2) - b);

cm1 = confusionmat(y,f(X))

% Versicolor vs others
alpha1 = zeros(150,1);
alpha1(I2) = 1/n2;
alpha2 = zeros(150,1);
alpha2(I1) = 1/n1;
alpha2(I3) = 1/n3;

K = X*X';
b = (alpha1'*K*alpha1 - alpha2'*K*alpha2)/2;

f = @(x) sign(sum(bsxfun(@times,alpha1-alpha2,K(:,x)),2) - b);

cm2 = confusionmat(y,f(X))

% Virginica vs others
alpha1 = zeros(150,1);
alpha1(I3) = 1/n3;  
alpha2 = zeros(150,1);
alpha2(I1) = 1/n1;
alpha2(I2) = 1/n2;

K = X*X';
b = (alpha1'*K*alpha1 - alpha2'*K*alpha2)/2;

f = @(x) sign(sum(bsxfun(@times,alpha1-alpha2,K(:,x)),2) - b);

cm3 = confusionmat(y,f(X))