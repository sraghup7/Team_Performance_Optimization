function score = label_fast_exact(my_data,LL,currentTeam,i0, prune)
% TEAMREP-FAST-APPROX algorithm for team replacement on player network
% 
% Inputs:
% my_data: the whole NBA network, e.g., player-playernumber of seasons together network
% LL: label matrix cell, e.g., if there are dn skills, then LL is a cell of size dn, 
%     LL{i} is a nxn diagonal matrix, LL{i}(j,j) shows the strength of j-th person
%     having i-th skill
% currentTeam: current players in the team.
% i0: the player to be replaced
% prune: prune or not?
%
% Output:
% score: each row is a score and its candidate id, note it's not sorted

% Set default value for prune
if nargin < 5
    prune = false;
end

n = size(my_data, 1); % number of players in the network
dn = length(LL); % number of skills

remainTeam = setdiff(currentTeam, i0); % current team without the player to be replaced
currentTeam = [remainTeam, i0]; % current team including the player to be replaced

W = my_data(currentTeam, currentTeam); % adjacency matrix of the current team
W = (triu(W, 1) + tril(W, -1)); % make the adjacency matrix symmetric

% W0 is the common part
n0 = length(currentTeam);
W0 = W;
W0(n0, 1:n0) = 0;
W0(1:n0, n0) = 0;

L = cell(1, dn);
for i = 1:dn
    L{i} = LL{i}(currentTeam, currentTeam); % label matrix of the current team
end

% L0 is the common part
L0 = cell(1, dn);
for i = 1:dn
    temp = L{i};
    temp(n0, n0) = 0;
    L0{i} = temp;
end

cand = setdiff((1:n), currentTeam); % candidate players for team replacement

% Prune unpromising candidates
if prune == true
    cand = cand(sum(my_data(cand, remainTeam), 2) > 0);
end

c = 0.00000001; % regularization parameter

q = ones(n0, 1) / n0;
p = ones(n0, 1) / n0;
qx = kron(q, q);
px = kron(p, p);

temp = zeros(n0 * n0);
for i = 1:dn
    temp = temp + kron(L{i} * W, L0{i} * W0); % combine the label and adjacency matrices
end
invZ = inv(eye(n0 * n0) - c * temp); % calculate the inverse of the matrix

R = zeros(n0 * n0, 1);
for i = 1:dn
    R = R + kron(L{i} * p, L0{i} * p); % calculate R
end

base = qx' * invZ * R; % calculate the base score
l = c * qx' * invZ;
r = invZ * R;

% Initialize a 2D array score with the same number of rows as the length of 
% the input array cand and two columns
score = zeros(length(cand), 2);

% Iterate over each element of the input array cand
for i = 1:length(cand)

    % Initialize vectors s and t
    s = [zeros(1, n0 - 1), 1];
    t = [my_data(cand(i), remainTeam),0];
    
    % Construct matrices A and B
    A = [t',s'];
    B = [s;t];   
    
    % Initialize vector a and cell array b
    a = [zeros(n0-1,1);1];
    b = cell(1,dn);
    
    % Fill cell array b with vectors
    for j = 1:dn
        b{j} = [zeros(1,n0-1),LL{j}(cand(i),cand(i))];
    end
    
    % Construct matrices A1 and B1
    A1 = [];
    B1 = [];
    for j = 1:dn
        A1 = [A1,kron(L{j},a)];
        B1 = [B1;kron(eye(n0),b{j})]; 
    end
    
    % Initialize matrices X1 and X2 with zeros
    X1 = zeros(n0*n0,n0*2);
    X2 = zeros(n0*n0,n0*2);
    
    % Calculate matrices X1 and X2
    for j = 1:dn
        X1 = X1 + kron(L{j}*W,L0{j}*A);
        X2 = X2 + kron(L{j}*W,a*b{j}*A);
    end
    
    % Calculate vectors Y1 and Y2
    Y1 = B1*kron(W,W0);
    Y2 = kron(eye(n0),B);
    
    % Construct matrices X and Y
    X = [A1,X1,X2];
    Y = [Y1;Y2;Y2];
    
    % Calculate matrix M
    M = inv(eye((dn+4)*n0)-c*Y*invZ*X);
    
    % Calculate vector r0
    r0 = invZ*A1*B1*px;
  
    % Calculate score for the i-th candidate team
    score(i,1) = base+qx'*r0+l*X*M*Y*(r+r0);
    score(i,2) = cand(i);
end
end
