% load('Data\DadesPlymouth\Signal\CombinedDataDade.mat')
% [imf_AD, imf_CR]=decompose_memd(data_AD, data_CR);
% if ~exist(['EnhData\Dataset_1\Signal\'],'dir')
%     mkdir(['EnhData\Dataset_1\Signal\'])
% end
% save('EnhData\Dataset_1\Signal\DecomposedIMFs_MEMD.mat', 'imf_AD', 'imf_CR');
% clc;clear;

load('Data\DadesPlymouth\Signal\CombinedDataDade.mat')
[imf_AD, imf_CR]=decompose_semd(data_AD, data_CR);
if ~exist(['EnhData\Dataset_1\Signal\'],'dir')
    mkdir(['EnhData\Dataset_1\Signal\'])
end
save('EnhData\Dataset_1\Signal\DecomposedIMFs_SEMD.mat', 'imf_AD', 'imf_CR');
clc;clear;

load('Data\DadesPlymouth\Signal\CombinedDataDade.mat')
[imf_AD, imf_CR]=decompose_cemd(data_AD, data_CR);
if ~exist(['EnhData\Dataset_1\Signal\'],'dir')
    mkdir(['EnhData\Dataset_1\Signal\'])
end
save('EnhData\Dataset_1\Signal\DecomposedIMFs_CEMD.mat', 'imf_AD', 'imf_CR');
clc;clear;

load('Data\MushaDatabase\Signal\CombinedDataMush.mat')
[imf_AD, imf_CR]=decompose_memd(data_AD, data_CR);
if ~exist(['EnhData\Dataset_2\Signal\'],'dir')
    mkdir(['EnhData\Dataset_2\Signal\'])
end
save('EnhData\Dataset_2\Signal\DecomposedIMFs_MEMD.mat', 'imf_AD', 'imf_CR');
clc;clear;

load('Data\MushaDatabase\Signal\CombinedDataMush.mat')
[imf_AD, imf_CR]=decompose_semd(data_AD, data_CR);
if ~exist(['EnhData\Dataset_2\Signal\'],'dir')
    mkdir(['EnhData\Dataset_2\Signal\'])
end
save('EnhData\Dataset_2\Signal\DecomposedIMFs_SEMD.mat', 'imf_AD', 'imf_CR');
clc;clear;

load('Data\MushaDatabase\Signal\CombinedDataMush.mat')
[imf_AD, imf_CR]=decompose_cemd(data_AD, data_CR);
if ~exist(['EnhData\Dataset_2\Signal\'],'dir')
    mkdir(['EnhData\Dataset_2\Signal\'])
end
save('EnhData\Dataset_2\Signal\DecomposedIMFs_CEMD.mat', 'imf_AD', 'imf_CR');
clc;clear;


function [imf_AD, imf_CR]=decompose_memd(data_AD, data_CR)
    imf_AD = MEMD(data_AD);
    imf_CR = MEMD(data_CR);
end
function [imf_AD, imf_CR]=decompose_semd(data_AD, data_CR)
    imf_AD = SEMD(data_AD);
    imf_CR = SEMD(data_CR);
end
function [imf_AD, imf_CR]=decompose_cemd(data_AD, data_CR)
    imf_AD = CEMD(data_AD);
    imf_CR = CEMD(data_CR);
end
function imf=MEMD(data)
    [n_trial, n_channel, n_sample] = size(data);
    data = reshape(permute(data, [2,3,1]), n_channel, n_trial * n_sample);
    imf=memd(data');
    imf=permute(imf, [2,1,3]);% n_imf, n_channel, n_sample*n_trial
    n_imf= size(imf,1);
    imf = reshape(imf, n_imf, n_channel, n_sample, n_trial);
    imf = permute(imf, [1,4,3,2]); %n_imf, n_trial, n_sample, n_channel
end
function imf=SEMD(data)
    [n_trial, n_channel, n_sample] = size(data);
    data = reshape(permute(data, [2,3,1]), n_channel, n_trial * n_sample);
    n_imf=15;
    imf=zeros(n_imf, n_channel, n_sample*n_trial);
    for i=1:n_channel
        [tmpimf, tmpres]=emd(data(i,:));
        tmpimf=cat(2,tmpimf,tmpres);
        n_tmpimf=size(tmpimf,2);
        if n_tmpimf<=n_imf
            imf(1:n_tmpimf,i,:)=tmpimf';
        else
            tmpimf(:,n_imf)=sum(tmpimf(n_imf:end),2);
            imf(:,i,:)=tmpimf';
        end
    end
    imf = reshape(imf, n_imf, n_channel, n_sample, n_trial);
    imf = permute(imf, [1,4,3,2]); %n_imf, n_trial, n_sample, n_channel
end
function imf=CEMD(data)
    [n_trial, n_channel, n_sample] = size(data);
    n_imf=15;
    imf=zeros(n_imf, n_channel, n_sample, n_trial);
    for i=1:n_trial
        for j=1:n_channel
            [tmpimf, tmpres]=emd(squeeze(data(i,j,:)));
            tmpimf=cat(2,tmpimf,tmpres);
            n_tmpimf=size(tmpimf,2);
            if n_tmpimf<=n_imf
                imf(1:n_tmpimf,j,:,i)=tmpimf';
            else
                tmpimf(:,n_imf)=sum(tmpimf(n_imf:end),2);
                imf(:,j,:,i)=tmpimf';
            end
        end
    end
    imf = permute(imf, [1,4,3,2]);
end