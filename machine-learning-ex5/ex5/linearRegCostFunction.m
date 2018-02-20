function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%first part of the equation
hTheta = X * theta;
diff = hTheta - y;
diff = diff .^ 2;
firstPart = (sum(diff(:))) / 2 / m;

% second part of the equation
thetaSquare = theta .^ 2;
sumThetaSquare = sum(thetaSquare(2:end));
secondPart = regularization = lambda / 2 / m * sumThetaSquare;


% cost function 
J = firstPart + secondPart;








% =========================================================================

grad = grad(:);

end
