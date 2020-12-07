function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% form X matrix with X_0 = 1
x1 = [ones(m,1), X];

% second layer(next to input) calc: z2 = x1*theta1, a2 = g(z2)

% X = 5000 x 400, x1 = 5000 x 401, theta1 = 25 x 401
z2 = x1 * Theta1'; % z2 = 5000 x 25
a2 = sigmoid(z2); % a2는 input layer/layer 1 => 2)에서 계산되어 나온 결과값들

% x2 = 5000 x (25+1), theta2 = 10 x 26
x2 = [ones(m,1), a2]; % make 5000 x 26 to calc forward from layer 2 => 3
z3 = x2 * Theta2'; % z3 = 5000 x 10
a3 = sigmoid(z3); % a3 = h_theta(x) 최종 레이어 => 이걸로 이제 하나의 값을 도출
% result => 5000 x 10

% reshape y to calc in Cost Function
y3 = zeros(m, num_labels); % init y3 as 5000 x 10
% each row represents y value(which is 1-10) that is vectorized
for i = 1:m
  y3(i, y(i)) = 1; % put 1 depending on y value (10 -> 0 0 0 0 0 0 0 0 0 1) (1 -> 1 0 0 0 0 0 0 0 0 0)
end

% calc Cost Function
% J = -(1/m) * sum1(sum2 logistic) + lambda/2m(sum1(sum2(sum3 theta^2)))
% y3 = 5000 x 10 a = 5000 x 10
J = (1/m)*sum(sum(-y3 .* log(a3) - (1-y3) .* log(1-a3)));
% REGULARIZE COST FUNCTION (DO NOT REGULARIZE 1st columns(BIAS UNIT))
J = J + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
for

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
