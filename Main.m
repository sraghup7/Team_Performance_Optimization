load network_modified;%loading NBA Network
currentTeam = [609,486,209,1010,597]; %selecting a team from the pool of players
i0= 486;%ID of player that needs to be replaced
count=csvread("result1.csv");%matrix representing the players and the position they played
dn=5; % number of positions
L = cell(1,dn);
for i=1:dn 
    L{i} = diag(count(:,i));
end

fileID=fopen('players.txt'); %this file contains look up dictionary for player IDs
Player_Dict=textscan(fileID,'%s','delimiter','\n');
Player_Dict=Player_Dict{1};

fprintf('We need to replace %s ...\n', Player_Dict{i0});
%Calling the fast-exact algorithm
tic
score = label_fast_exact(my_data,L,currentTeam,i0,true);
%the score is calculated using svd
top5 = topfive(score);
display 'Using TEAMREP-FAST-EXACT, the top five candidates are:';
fprintf('%s \n', Player_Dict{top5});
fprintf('%s', 'time taken');
disp(toc)
%Calling the fast-approx algorithm
tic
score = label_fast_approx(my_data,L,currentTeam,i0,true,2);
%the score is calculated using an inverse matrix
top5 = topfive(score);
display 'Using TEAMREP-FAST-APPROX, the top five candidates are:';
fprintf('%s \n', Player_Dict{top5});
fprintf('%s', 'time taken');
disp(toc)
