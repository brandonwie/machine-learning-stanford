function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta)); % grad의 사이즈는 (n+1) x 1

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% X => m(데이터수) x (n+1) // theta => (n+1) x 1 :  여기서 n+1은 column수 즉, 3
% X의 경우 coloum이 n개 이지만 앞서 ex2.m에서 1을 m row 만큼 first column에 붙임
% 따라서 변환없이 곱 가능 : 출력값 m x 1
h = sigmoid(X*theta);

J = -(1/m)*(y'*log(h)+(1-y)'*log(1-h));

grad = (1/m)*X'*(h-y);
% y는 m x 1 행렬

% =============================================================

end
