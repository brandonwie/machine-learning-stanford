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

% form X matrix with X_0 = 1 ; activation 1
a1 = [ones(m,1), X];
% second layer(next to input) calc: z2 = a1*theta1, a2 = g(z2)
% X = 5000 x 400, x1 = 5000 x 401, theta1 = 25 x 401
z2 = a1 * Theta1' ; % z2 = 5000 x 25
a2 = sigmoid(z2); % a2는 input layer/layer 1 => 2)에서 계산되어 나온 결과값들

% a2 = 5000 x (25+1), theta2 = 10 x 26
a2 = [ones(size(a2, 1),1),  a2]; % make 5000 x 26 to calc forward from layer 2 => 3
z3 = a2 * Theta2' ; % z3 = 5000 x 10
a3 = sigmoid(z3); % a3 = h_theta(x) 최종 레이어 => 이걸로 이제 하나의 값을 도출
hx = a3;
% result => 5000 x 10

y_Vec = (1:num_labels) == y; % 5000 x 10
% calc Cost Function
% J = -(1/m) * sum1(sum2 logistic) + lambda/2m(sum1(sum2(sum3 theta^2)))
% y3 = 5000 x 10 a = 5000 x 10
J = (1/m)*sum(sum(-y_Vec .* log(hx) - (1-y_Vec) .* log(1-hx))); % scalar
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
for t = 1:m
  % forward propagation
  % shape a_1 as column vector
  a_1 = X(t,:)'; % take respective x values apply to first activation val
  a_1 = [1; a_1]; % add 1 at the front to ensure that the activation vectors include the bias unit

  z_2 = Theta1 * a_1; % theta*x
  a_2 = sigmoid(z_2); % get activation value
  a_2 = [1; a_2]; % add bias unit

  z_3 = Theta2 * a_2; % 10 x 1
  a_3 = sigmoid(z_3); % 10 x 1 last layer


  % % compute error / shape y_3 as column vector
  % y_3 = y3(t,:)';
  yVector = (1:num_labels)'==y(t); % 10 x 1
  % calc delta 3(error 3)
  d_3 = a_3 - yVector;
  % error 2
  d_2 = (Theta2' * d_3) .* sigmoidGradient([1; z_2]); % (hidden_layer_size + 1) x 1 == 26 x 1
  % remove delta_0
  d_2 = d_2(2:end);

  % Delta^l = Delta^l + delta^(l+1)*a^l'
  Theta1_grad = Theta1_grad + (d_2 * a_1'); % 25 x 401
  Theta2_grad = Theta2_grad + (d_3 * a_2'); % 10 x 26

end

% now divide accumulated Delta by m
Theta2_grad = (1/m) * Theta2_grad;
Theta1_grad = (1/m) * Theta1_grad;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
