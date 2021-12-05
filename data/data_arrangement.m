% data spiltting 
% 50 slices for testing. 30% for training and 20% validation
clc
clear

load training_data_raw

paraNum = size(result,3);
gridNum = size(result,1);

IndexList = 1:paraNum;

% testingIndex = randperm(paraNum,paraNum*testingPerc);
testingIndex = randperm(paraNum,50);

trainingIndex = setdiff(IndexList,testingIndex);

testingData = result(:,:,testingIndex);
testingPara = paralist(testingIndex,:);

trainingData = result(:,:,trainingIndex);
trainingPara = paralist(trainingIndex,:);

[x,t] = meshgrid(linspace(-1,1,1000),linspace(0,1,1000));
t = t';
x = flipud(x');

x = reshape(x,[gridNum*gridNum,1]);
t = reshape(t,[gridNum*gridNum,1]);

data = [];
for i = 1:size(trainingIndex,2)
    i
    temp(:,1) = x;
    temp(:,2) = t;
    temp(:,3) = repmat(trainingPara(i,1),[gridNum*gridNum,1]);
    temp(:,4) = repmat(trainingPara(i,2),[gridNum*gridNum,1]);
    temp(:,5) = reshape(trainingData(:,:,i),[gridNum*gridNum,1]);
%     sampleIndex = randperm(gridNum*gridNum,gridNum*gridNum*0.3);
%     temp2 = temp(sampleIndex,:);
    data = [data;temp];

end

sampleIndex = randperm(size(data,1),size(data,1)*0.5);
% sampleRestIndex = setdiff(1:size(data,1),sampleIndex);

data_train_val = data(sampleIndex,:);
sampleIndex_val = randperm(size(data_train_val,1),size(data_train_val,1)*0.2);
sampleIndex_train = setdiff(1:size(data_train_val,1),sampleIndex_val);

data_val = data_train_val(sampleIndex_val,:);
data_train = data_train_val(sampleIndex_train,:);

save('testing_index.mat','testingIndex');
save('train_val_index.mat','sampleIndex');
save('train_index.mat','sampleIndex_train');

save('train_dataset_two.mat','data_train');
save('val_dataset_two.mat','data_val');