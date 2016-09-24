function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
numberOfFeatures = size(X, 2);
numberOfRows = size(X, 1);
thetaTemp = theta;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    % Wichtig: Ich muss jedes Theta abfangen

    %prediction = (theta(1) + (theta(2) * x));
    prediction = X * theta; % correct

    errorFunction = prediction - y;
    for j = 1:numberOfFeatures
      if j == 1,
        thetaTemp(j) = theta(j) - alpha * (1/m) * sum(errorFunction);
      else
        Xtemp = errorFunction' * X(:,j);
        thetaTemp(j) = theta(j) - alpha * (1/m) * sum(Xtemp);
      end

    theta = thetaTemp;



    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
