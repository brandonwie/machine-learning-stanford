function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations.
%
% Hint: When computing the gradient of the regularized cost function,
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta;
%           temp(1) = 0;   % because we don't add anything for j = 0
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
% hypothesis는 이전 ex2와 마찬가지로 계산
% X_t = 5x4, theta = 4x1 => 그냥 곱해도 됨
h = sigmoid(X * theta);
% theta_0의 값은 0으로 설정 / 왜 ex3.m의 theta^4x1 (1,1) 값이 -2로 설정되었는지 모르겠음
theta(1,1) = 0;
J = (1/m) * ((-y)' * log(h) - (1-y)' * log(1-h)) + (lambda/(2*m)) * (theta' * theta);
grad = (1/m) * X' * (h-y) + (lambda/m) * theta;
% 1.3.3 : j = 0 일때, grad(1,1)의 식은 lamba를 포함하지 않는다.
grad(1,1) = (1/m) * X'(1,:) * (h-y);

% =============================================================
% vector  mx1 로 만든다.
grad = grad(:);

end
