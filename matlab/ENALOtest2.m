function [risk] = ENALOtest2(X, y, alpha, lambda)
%Proximal method
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
        betaA = beta1(A, i);
        J = zeros(k, 1);        
        L1 = (betaA > alpha * lambda(i));
        L1 = logical(L1);
        L2 = (betaA < -alpha * lambda(i));
        L2 = logical(L2);
        J(L1) = 1 / (1 + (1 - alpha) * lambda(i));
        J(L2) = 1 / (1 + (1 - alpha) * lambda(i));
        %J(L1) * betaA = betaA(L1) / (1 + (1 - alpha) * lambda(i)) - alpha * lambda(i) / (1 + (1 - alpha) * lambda(i));
        %J(L2) = betaA(L2) / (1 + (1 - alpha) * lambda(i)) + alpha * lambda(i) / (1 + (1 - alpha) * lambda(i));
        J = diag([0;J]);
        H =  XA * pinv(J * (XA' * XA) + eye(k + 1) - J) * J * XA';
        risk(i) = norm((eye(n) + diag(diag(H) ./ (1 - diag(H)))) * diff(:, i)) / n; 
    else
        risk(i) = norm(diff(:, i)) / n;
    end
end
end

