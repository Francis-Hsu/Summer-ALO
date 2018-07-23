%%glmnet package needed
%%addpath(''); 
rng('default')

%data initialization
n = 300;
p = 600;
k = 60;
beta = zeros(p, 1);
t = randsample(p, k);
beta(t) = normrnd(0, 1, [k, 1]);
loglambda = linspace(log(1E-4), log(1E-2), 25);
lambda = exp(loglambda); 
ntimelambda = lambda * n; %because the formula in glmnet use 1 / 2n

%missepcification case
X1 = normrnd(0, sqrt(1/k), [n, p]);
epsilon1 = normrnd(0, 0.5, [n, 1]);
y1 = X1 * beta + epsilon1;
y1(y1>0) = sqrt(y1(y1>0));
y1(y1<0) = -sqrt(-y1(y1<0));

%test for alpha = 0.1
opt1.alpha = 0.1;
opt1.lambda = lambda;
opt1.standardize = false;
opt1.intr  = true;

%True
diffLOO = zeros(n, 25);
for j = 1:n
    XLOO = X1;
    yLOO = y1;
    XLOO(j,:) = 0;
    yLOO(j) = 0;
    solstruct = glmnet(XLOO, yLOO, [], opt1);
    solLOO = solstruct.beta;
    solinter = solstruct.a0;
    diffLOO(j, :) = y1(j) - X1(j, :) * solLOO - solinter';
    disp(j);
end
truerisk = sqrt(diag(diffLOO' * diffLOO)) / n;
%may meet error when running glmnet on win 10

%Primal
risk1 = ENALOtest(X1, y1, 0.1, ntimelambda);
plot(log(ntimelambda), truerisk(25:-1:1));
plot(log(ntimelambda), risk1(25:-1:1),log(ntimelambda), truerisk(25:-1:1));

%Proximal
risk2 = ENALOtest2(X1, y1, 0.1, ntimelambda);
plot(log(ntimelambda), risk2(25:-1:1), log(ntimelambda), truerisk(25:-1:1));

save('risktrue', 'truerisk');
