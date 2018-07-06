function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Debug code 
% load('ex7data2.mat');
% centroids = [3 3; 6 2; 8 5];

% Set K
% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

distance = zeros(size(X,1), length(centroids));

for i = 1:length(centroids)
  % diff = bsxfun(@minus, X, centroids(i, : ));
  % distance = sum(diff .^ 2, 2);
  diff = X - centroids(i, :);
  diffSquared = diff .^ 2;
  diffSquaredSum = sum(diffSquared, 2);
  
  distance(:, i) = diffSquaredSum;
endfor

[temp, idx] = min(distance, [], 2);


% =============================================================

end