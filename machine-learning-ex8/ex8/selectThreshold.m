function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

F11 = [];

stepsize = (max(pval) - min(pval)) / 1000;

epsilon1 = min(pval):stepsize:max(pval);

for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

  truePositive = 0; 
  trueNegative = 0;
  falsePositive = 0; 
  falseNegative = 0;  
    
  % prediction for the particular epsilon  
  predictions = pval < epsilon;  
  
  for i = 1:size(yval,1)
    if(yval(i) == 1 && predictions(i) == 1)
      truePositive++;
    endif  
    if(yval(i) == 0 && predictions(i) == 1)
      falsePositive++;
    endif  
    if(yval(i) == 0 && predictions(i) == 0)
      trueNegative++;
    endif  
    if(yval(i) == 1 && predictions(i) == 0)
      falseNegative++;
    endif  
  endfor
    
    
    
  precision = truePositive / (truePositive + falsePositive);
  recall = truePositive / (truePositive + falseNegative);
  
  F1 =  2 * precision * recall / (precision + recall);
  F11 = [F11; F1];
  
    % =============================================================    
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
       fprintf('F1 on Cross Validation Set:  %f\n', F1);
    end
end

end
