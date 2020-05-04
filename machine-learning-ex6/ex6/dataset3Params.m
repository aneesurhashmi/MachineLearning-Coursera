function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%



C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
err = 1;

model_c = zeros(1,1);
params = zeros(3,1);

for i = 1: length(C_vec)
	for j = 1:length(sigma_vec)
		model= svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j)));
		fprintf('Error.\n');
		%prediction = Xval*model.w
		prediction = svmPredict(model, Xval);
		err_i = 1 - mean(double(prediction == yval));	
		if err_i < err
			[i,j]
			params(1) = C_vec(i)
			params(2) = sigma_vec(j)		
			model_c = model;			
			err = err_i
		end
	end
end


C = params(1);
sigma = params(2);

%=========================================================================

end
