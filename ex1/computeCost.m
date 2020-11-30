function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

hypothesis = X*theta; % (m*2)*(2*1) = m*1 (가설함수 h(x)값)
sqrErrors = (hypothesis - y).^2; % 인구 수*계수(theta) (가설함수h(x))에서 실제 profit을 빼준 값

J = 1/(2*m) * sum(sqrErrors); % mean squared errors

% =========================================================================

end
