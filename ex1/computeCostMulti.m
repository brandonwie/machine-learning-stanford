function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h = X*theta; % (m*3)*(3*1) = m*1 (가설함수 h(x)값)
errSqr = (h - y).^2; % 집값*계수(theta) = 가설함수h(x))에서 실제 profit을 빼준 값
J = 1/(2*m) * sum(errSqr); % mean squared errors

% =========================================================================

end
