% Data set generator with Allen Cahn equation 

clc
clear

meshDensity = 1000;

a1Set = linspace(1e-4,1e-1,10);

a2Set = linspace(1,10,10);
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

for i = 1:size(computationlist,1)
    tic
    a1 = computationlist(i,1);
    a2 = computationlist(i,2);
    a3 = a2;
        
    [u,x,t] = AllenEQ(a1,a2,a3,meshDensity);
    
     u = reshape(u,[1000*1000,1]);
     u = u';
    
%     result_file_name = ['/Users/huangchenchen/Dropbox/AME508_Project/data_generating/observation_data/',num2str(i)];
%     result_m = matfile(result_file_name,'writable',true);
% % 
%     result_m.u = u;
    
    save(['/Users/huangchenchen/Dropbox/AME508_Project/data_generating/observation_data/',num2str(i)],'u','-v7')
    
    toc
end
