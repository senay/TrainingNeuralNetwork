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

% Add ones to the X data matrix
X = [ones(m, 1) X];


%[mx p] = max(h_theta); %%%% index is equivalent to the label
%p = p';


%size(X)
%max(y)
cost = 0;
for i = 1:m
    yy = zeros(max(y),1);
    a_1 = X(i,:);
    a_2 = sigmoid(Theta1*a_1');

    % Add ones to the X data matrix
    a_2 = a_2';
    a_2 = [ones(size(a_2,1), 1) a_2];

    a_3 = sigmoid(Theta2*a_2');
    h_theta = a_3;

    yy(y(i,1),1)=1;
    cost = cost + (-yy'*log(h_theta)-(1-yy')*log(1-h_theta));
end


J = cost/m + (sum(sum(Theta1.*Theta1)) + sum(sum(Theta2.*Theta2)))*lambda/(2*m);


%
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
%
Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));
for t=1:m
    yy =zeros(num_labels,1);
    a_1 = X(t,:); 
    a_2 = sigmoid(Theta1*a_1');
    % Add ones to the X data matrix
    a_2 = a_2';
    a_2 = [ones(size(a_2,1), 1) a_2];
    a_3 = sigmoid(Theta2*a_2');
    
    yy(y(t,1),1)=1;
    delta_3 = a_3 - yy;
    aa = Theta2'*delta_3;
    delta_2 = aa(2:end).*sigmoidGradient(Theta1*a_1');

    Delta_1 = Delta_1 + delta_2*a_1 ;   
    Delta_2 = Delta_2 + delta_3*a_2;  
end


reg1 = Theta1*(lambda/m);
reg2 = Theta2*(lambda/m);

reg1(:,1) = zeros(size(reg1,1),1);
reg2(:,1) = zeros(size(reg2,1),1);

Theta1_grad = Delta_1/m + reg1;
Theta2_grad = Delta_2/m + reg2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% -------------------------------------------------------------

% =========================================================================

end
