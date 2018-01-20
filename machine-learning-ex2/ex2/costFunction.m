function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

thetaTransposeX = X * theta;
% sigmoidThetaTransposeX is the hypothesis function 
sigmoidThetaTransposeX = sigmoid(thetaTransposeX);
firstPart = -y .* log(sigmoidThetaTransposeX);
secondPart = (1 - y) .* log(1 - sigmoidThetaTransposeX);

forEachRow = firstPart - secondPart;

allRowSummation = sum(forEachRow);

J = allRowSummation / m;


numOfFeature = columns(X);
for jIndex = 1 : numOfFeature
  grad(jIndex) = sum((sigmoidThetaTransposeX - y) .* X(:, jIndex)) / m;
endfor;


% =============================================================

end
