function [theta, J_history] = gradientDescentMulti(cost_func, nn_params, architecture, X, y, alpha,lambda, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
input_layer_size =  architecture(1);
hidden_layer_size =  architecture(2);
num_labels =  architecture(3);
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);



% testing 


theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));



for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
	z = sum(theta1'.*X,2); % multiply theta with respective feature and then add rows
	% sigmoid function
	h = 1./(1+e^(-z));
	theta1 = theta1 - alpha*(1/m)*sum((h-y).*X)';

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = cost_func(X, y, theta);
	cost_func(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambda);
end
end
