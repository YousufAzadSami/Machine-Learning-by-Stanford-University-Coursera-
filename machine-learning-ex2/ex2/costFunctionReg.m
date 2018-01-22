function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% get the cost
thetaTransposeX = X * theta;
% sigmoidThetaTransposeX is the hypothesis function 
sigmoidThetaTransposeX = sigmoid(thetaTransposeX);
firstPart = -y .* log(sigmoidThetaTransposeX);
secondPart = (1 - y) .* log(1 - sigmoidThetaTransposeX);

forEachRow = firstPart - secondPart;
allRowSummation = sum(forEachRow);

costWithoutRegularization = allRowSummation / m;
thetaSumSquared = sum(theta(2:end) .^ 2);
penalty = lambda / 2 / m * thetaSumSquared;
penalty1 = lambda / 2 / m * (sum(theta(2:end)));  % debug code

J = costWithoutRegularization + penalty;

numOfFeature = columns(X);
for jIndex = 1 : numOfFeature
  grad(jIndex) = sum((sigmoidThetaTransposeX - y) .* X(:, jIndex)) / m;
  
  if(jIndex != 1)
    grad(jIndex) = grad(jIndex) + (lambda / m * grad(jIndex));  
  endif
  
endfor;

% get the gradient


% =============================================================

end
