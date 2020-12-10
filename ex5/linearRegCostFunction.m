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

% X = 12 x 2, y = 12 x 1, theta = 2 x 1
h = X*theta;
theta_r = [0; theta(2:end);]; % remove theta_0
J = (1/(2*m))*sum((h-y).^2) + (lambda/(2*m))*(theta_r'*theta_r);
grad = (1/m)*(X'*(h - y)) + (lambda/m)*theta_r; % grad = 2 x 1


% =========================================================================

grad = grad(:);

end