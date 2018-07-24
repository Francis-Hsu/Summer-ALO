function [risk] = ENALOtest(X, y, alpha, lambda)
%Primal method
% test version

[n, ~] = size(X);
lambda = sort(lambda, 'descend');
nlambda = length(lambda);
risk = zeros(nlambda, 1);

opt1.alpha = alpha;
opt1.lambda = lambda;
opt1.standardize = false;
opt1.intr  = true;

sol1 = glmnet(X, y, [], opt1);
intercept = sol1.a0;
beta1 = sol1.beta;
diff = intercept' + X * beta1 - y;

for i = 1:nlambda
    A = (abs(beta1(:, i)) > 1e-8); 
    A = logical(A); 
    k = sum(A);
    if k > 0
        XA = [ones(n, 1), X(:, A)];
        H =  XA * pinv(XA' * XA + diag([0; lambda(i) * (1 - alpha) * ones(k, 1)])) * XA';
        risk(i) = norm((eye(n) + diag(diag(H) ./ (1 - diag(H)))) * diff(:, i)) / n; 
    else
        risk(i) = norm(diff(:, i)) / n;
    end
end
end

