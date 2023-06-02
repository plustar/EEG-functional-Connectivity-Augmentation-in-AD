EMDType={'MEMD'};Classifier={'ResNet18_lr4','BrainNet_lr4_2','EEGNet_lr4_2'};
NENH=[0,5,10,15,20,25,30,35,40,45,50,100,150,200,250,300,350,400,450,500];
acc=zeros(2,length(Classifier),length(EMDType),10,length(NENH),200);

n_test=[21, 40];
for dataset=1:2
    for cls=1:3%length(Classifier)
        for emd=1:length(EMDType)
%             if cls==4
%                 load(['Model/Dataset_',num2str(dataset),'/',EMDType{emd},'/',Classifier{cls},'/info.mat'])
%                 acc(dataset,cls,emd,:,1:length(NENH))=accuracy;
%                 continue
%             end
            for n=1:10
                
                load(['Model/Dataset_',num2str(dataset),'/',EMDType{emd},'/',Classifier{cls},'/info_fold',num2str(n),'_nenh500.mat'])
                if cls<3
                    correct=reshape(correct,100,[]);
                    acc(dataset,cls,emd,n,1:length(NENH),1:100)=double(correct(1:100,:))'/n_test(dataset);
                else
                    correct=reshape(correct,200,[]);
                    acc(dataset,cls,emd,n,2:length(NENH),:)=double(correct(1:200,2:end))'/n_test(dataset);
                    load(['Model/Dataset_',num2str(dataset),'/',EMDType{emd},'/',Classifier{cls},'/info_fold',num2str(n),'_nenh0.mat'])
                    acc(dataset,cls,emd,n,1,:)=double(correct(:))./n_test(dataset);
                end
%                 if size(correct,2)==19
%                     acc(dataset, cls,emd,n,2:length(NENH),:)=double(correct(1:100,:))'/n_test(dataset);
%                     load(['Model/Dataset_',num2str(dataset),'/',EMDType{emd},'/',Classifier{cls},'/info_fold',num2str(n),'_nenh0.mat'])
%                     acc(dataset,cls,emd,n,1,:)=double(correct(1:100))/n_test(dataset);
%                 else
%                     if size(correct,2)==20
                    
%                     end
%                 end
            end
        end
    end
end

Accuracy=acc;
save F_accuracy.mat Accuracy EMDType Classifier
macc=squeeze(mean(acc,4));
sacc=squeeze(std(acc,0,4));
figure('Position',[200,200,1200,600])
for i=1:2
    for j=1:3
        subplot(2,3,(i-1)*3+j)
        plot(squeeze(macc(i,j,2,[1,5,10,15,20],:))')
        legend('location','best')
%         if i==1
%             axis([-inf,inf,0.4,0.85])
%         else
%             axis([-inf,inf,0.2,0.7])
%         end
    end
end