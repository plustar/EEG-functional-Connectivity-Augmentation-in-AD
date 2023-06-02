EMDType={'MEMD'};Classifier={'ResNet18_lr4_conf'};
NENH=[50];
acc=zeros(2,length(Classifier),length(EMDType),10,length(NENH),200);

n_test=[21, 40];
for dataset=1:2
    for cls=1:1%length(Classifier)
        for emd=1:length(EMDType)
            for n=1:10
                load(['Model/Dataset_',num2str(dataset),'/',EMDType{emd},'/',Classifier{cls},'/info_fold',num2str(n),'_nenh0.mat'])
                acc(dataset,cls,emd,n,1)=double(correct(end))/n_test(dataset);
                load(['Model/Dataset_',num2str(dataset),'/',EMDType{emd},'/',Classifier{cls},'/info_fold',num2str(n),'_nenh500.mat'])
            end
        end
    end
end