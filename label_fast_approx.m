function score = label_fast_approx(my_data,LL,currentTeam,i0,prune,r)
% This function implements TEAMREP-FAST-APPROX for team replacement on player network
% Input:
%   my_data: the whole NBA network, e.g., player-playernumber of seasons together network
%   L: label matrice cell, e.g., if there are dn skills, then L is a cell of size dn,
%      L{i} is a nxn diagonal matrix, L{i}(j,j) shows the strength of j-th person having i-th skill
%   currentTeam: current players in the team.
%   i0: the player to be replaced
%   prune: prune or not? (default is false)
%   r: the approximate rank (default is floor(sqrt(n0)))
% Output:
%   score: each row is a score and its candidate id, note it's not sorted

if nargin < 4
    prune = false; 
    % If the function is called with only 3 arguments, set the prune flag to false by default
end
n=size(my_data,1);
remainTeam = setdiff(currentTeam,i0);
% Get the current team without the player to be replaced
currentTeam = [remainTeam, i0];
% Add the player to be replaced back to the current team

n0 = length(currentTeam);
W0 = zeros(n0,n0);
w_temp = my_data(remainTeam,remainTeam); 
% Get the adjacency matrix for the current team
w_temp = (triu(w_temp,1) + tril(w_temp,-1)); 
% Remove the diagonal elements from the adjacency matrix
W0(1:n0-1,1:n0-1) = w_temp;
% Fill in the adjacency matrix for the current team

if nargin < 6
    r = floor(sqrt(n0)); % If the function is called with only 5 arguments, set the rank to floor(sqrt(n0)) by default
end

% top r eigen-decomposition
[U0,Lam0,~]=svds(W0,r,'L');
% Compute the top r left singular vectors and singular values for the current team's adjacency matrix
U = U0;
V = Lam0*U0'; 
% Compute the top r right singular vectors for the current team's adjacency matrix
s = [zeros(n0-1,1);1]; 
% Add a row and column of zeros and a final element of 1 to the right singular vectors for use in calculations

w_original = my_data(i0,currentTeam);
% Get the row in the adjacency matrix corresponding to the player to be replaced
X = [U,w_original', s];
% Concatenate the left singular vectors with the player to be replaced and the zeros and 1 from s
Y = [V;s';w_original];
% Concatenate the right singular vectors with the zeros and 1 from s and the row for the player to be replaced

cand = setdiff((1:n),currentTeam);
% Get the set of candidates by removing the current team from the set of all players
if prune == true
    cand = cand(sum(my_data(cand,remainTeam),2)>0); 
    % If the prune flag is true, remove candidates that don't have any connections with the current team
end

c=0.00000001; 
% Set the regularization constant

dn=length(LL); 
% Get the number of skills
% Initialize q and p with the uniform distribution
q = {ones(n0,1)/n0,ones(n0,1)/n0};
p = {ones(n0,1)/n0,ones(n0,1)/n0};
p1 = p{1};
p2 = p{2};
q1 = q{1};
q2 = q{2};
score = zeros(length(cand), 2);
for i=1:length(cand)
    % add candidate player to the team
    newTeam = [remainTeam,cand(i)];
    
    % calculate the weight matrix between the new team and all other players
    w_new = my_data(cand(i),newTeam);
    
    % construct the X and Y matrices
    X_new = [U, w_new', s];
    Y_new = [V;s';w_new];

    % construct the L1 and L2 matrices for each skill
    for j=1:dn
        L1{j} = LL{j}(currentTeam,currentTeam);
        L2{j} = LL{j}(newTeam,newTeam);
    end
 
    % calculate the approximation of the inverse of tilde(Lam)
    temp  = zeros((r+2)^2,(r+2)^2);
    for j=1:dn
        temp = temp + kron(Y*L1{j}*X,Y_new*L2{j}*X_new);
    end
    tildeLam = inv( eye((r+2)^2) - c*temp);

    % calculate the score of the candidate player
    L = zeros(1,(r+2)^2);
    R = zeros( (r+2)^2,1);
    base = 0;
    for j=1:dn
        L = L + kron(q1'*L1{j}*X,q2'*L2{j}*X_new);
        R = R + kron(Y*L1{j}*p1,Y_new*L2{j}*p2);
        base = base + (q1'*L1{j}*p1)*(q2'*L2{j}*p2);
    end
    score(i,1) = base + c*L*tildeLam*R; 
    % add the candidate score to the score matrix
    score(i,2) = cand(i);
    % add the candidate player id to the score matrix
end

end