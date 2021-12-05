% Data set generator with Allen Cahn equation 

clc
clear

meshDensity = 1000;

a1Set = linspace(1e-4,1e-1,15);

a2Set = linspace(1,10,15);
% a2Set = 5;

a3Set = a2Set;

%% Create list
computationlist = [];
computationlistnum = 1;
for i = 1:numel(a1Set)
    for j = 1:numel(a2Set)
           computationlist(computationlistnum,:) = [a1Set(i) a2Set(j)];
           computationlistnum = computationlistnum + 1;
    end
end

uData = [];
xgrid = [];
tgrid = [];

parfor i = 1:size(computationlist,1)
    
    a1 = computationlist(i,1);
    a2 = computationlist(i,2);
    a3 = a2;
        
    [u,x,t] = AllenEQ(a1,a2,a3,meshDensity);
    
    uData(:,:,i) = u';

end

result_file_name = 'training_data_raw';
result_m = matfile(result_file_name,'writable',true);
% 
result_m.result = uData;
result_m.paralist = computationlist;
% result_m.xgrid = xgrid;
% result_m.tgrid = tgrid;
