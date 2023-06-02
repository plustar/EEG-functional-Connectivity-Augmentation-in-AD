using MAT
using Random
using DataFrames

function generate_index(data_AD,data_CR,iters)
    n_ADs = size(data_AD)[1]
    n_CRs = size(data_CR)[1]
    randind=randperm(MersenneTwister(iters), n_ADs)
    train_ind_AD=randind[ 1: 10]
    test_ind_AD =randind[11:end]
    randind=randperm(MersenneTwister(iters+10), n_CRs)
    train_ind_CR=randind[ 1: 10]
    test_ind_CR =randind[11:end]
    return train_ind_AD, test_ind_AD, train_ind_CR, test_ind_CR
end

function generate_realcols(data, train_ind, test_ind)
    train    =data[train_ind,:,:]
    test     =data[test_ind,:,:]
    return train, test
end
function generate_artifact(imf_ADs,train_ind_AD, iters, seed)
    N_ENHANCE_TRIAL = 500
    n_imf_AD, n_AD, n_sample, n_channel = size(imf_ADs)
    imf_AD      =imf_ADs[:,train_ind_AD,:,:]
    rInd_AD = zeros(Int64, N_ENHANCE_TRIAL, n_channel, n_imf_AD)
    n_AD = size(imf_AD)[2]
    println("$(iters): Generating random permutation ...")
    for i in 1:N_ENHANCE_TRIAL
        for j in 1:21
            rInd_AD[i,j,:] = rand(MersenneTwister((i - 1) * N_ENHANCE_TRIAL * 2 + j * 2 + seed), 1:n_AD, n_imf_AD)
        end
    end

    train_AD_imf_enh = zeros(N_ENHANCE_TRIAL, n_channel, n_imf_AD, n_sample)

    println("$(iters): Concatenating random selected IMFs ...")
    for i in 1:N_ENHANCE_TRIAL
        for j in 1:n_channel
            for k in 1:n_imf_AD
                train_AD_imf_enh[i,j,k,:] = imf_AD[k,rInd_AD[i,j,k],:,j]
            end
        end
    end

    println("Building enhanced signals ...")
    train_AD_enh = sum(train_AD_imf_enh, dims=3)[:,:,1,:];
    return train_AD_enh
end

# matwrite("Data/DadesPlymouth/AugumentData/enhd_Data_$(iters).mat",Dict("train_AD_enh" => train_AD_enh,"train_CR_enh" => train_CR_enh,
#                             "train_AD"    => train_AD,     "train_CR"  => train_CR,
#                             "test_AD"    => test_AD,       "test_CR"  => test_CR))
function generate_artifact_dataset1()
    data = matread("Data/DadesPlymouth/Signal/CombinedDataDade.mat")
    data_AD = data["data_AD"]
    data_CR = data["data_CR"]

    imfs = matread("EnhData/Dataset_1/Signal/DecomposedIMFs_MEMD.mat")
    imf_AD_MEMD  = imfs["imf_AD"]
    imf_CR_MEMD  = imfs["imf_CR"]

    imfs = matread("EnhData/Dataset_1/Signal/DecomposedIMFs_SEMD.mat")
    imf_AD_SEMD  = imfs["imf_AD"]
    imf_CR_SEMD  = imfs["imf_CR"]

    imfs = matread("EnhData/Dataset_1/Signal/DecomposedIMFs_CEMD.mat")
    imf_AD_CEMD  = imfs["imf_AD"]
    imf_CR_CEMD  = imfs["imf_CR"]
    try
        mkdir("EnhData/Dataset_1/SplitData/")
    catch
    end
    for i in 1:10
        train_ind_AD, test_ind_AD, train_ind_CR, test_ind_CR=generate_index(data_AD,data_CR,i)
        train_AD, test_AD=generate_realcols(data_AD, train_ind_AD, test_ind_AD)
        train_CR, test_CR=generate_realcols(data_CR, train_ind_CR, test_ind_CR)

        train_AD_enh=generate_artifact(imf_AD_MEMD,train_ind_AD,i,0);
        train_CR_enh=generate_artifact(imf_CR_MEMD,train_ind_CR,i,1);
        try
            mkdir("EnhData/Dataset_1/SplitData/MEMD")
        catch
        end
        file_memd="EnhData/Dataset_1/SplitData/MEMD/enhd_Data_$(i).mat"
        dict_memd=Dict("train_AD_enh" => train_AD_enh,"train_CR_enh" => train_CR_enh,
                        "train_AD"    => train_AD,     "train_CR"  => train_CR,
                        "test_AD"    => test_AD,       "test_CR"  => test_CR)
        matwrite(file_memd,dict_memd)
       

        train_AD_enh=generate_artifact(imf_AD_SEMD,train_ind_AD,i,0);
        train_CR_enh=generate_artifact(imf_CR_SEMD,train_ind_CR,i,1);
        try
            mkdir("EnhData/Dataset_1/SplitData/SEMD")
        catch
        end
        file_semd="EnhData/Dataset_1/SplitData/SEMD/enhd_Data_$(i).mat"
        dict_semd=Dict("train_AD_enh" => train_AD_enh,"train_CR_enh" => train_CR_enh,
                        "train_AD"    => train_AD,     "train_CR"  => train_CR,
                        "test_AD"    => test_AD,       "test_CR"  => test_CR)
        matwrite(file_semd,dict_semd)

        train_AD_enh=generate_artifact(imf_AD_CEMD,train_ind_AD,i,0);
        train_CR_enh=generate_artifact(imf_CR_CEMD,train_ind_CR,i,1);
        try
            mkdir("EnhData/Dataset_1/SplitData/CEMD")
        catch
        end
        file_cemd="EnhData/Dataset_1/SplitData/CEMD/enhd_Data_$(i).mat"
        dict_cemd=Dict("train_AD_enh" => train_AD_enh,"train_CR_enh" => train_CR_enh,
                        "train_AD"    => train_AD,     "train_CR"  => train_CR,
                        "test_AD"    => test_AD,       "test_CR"  => test_CR)
        matwrite(file_cemd,dict_cemd)
    end
end

function generate_artifact_dataset2()
    data = matread("Data/MushaDatabase/Signal/CombinedDataMush.mat")
    data_AD = data["data_AD"]
    data_CR = data["data_CR"]

    imfs = matread("EnhData/Dataset_2/Signal/DecomposedIMFs_MEMD.mat")
    imf_AD_MEMD  = imfs["imf_AD"]
    imf_CR_MEMD  = imfs["imf_CR"]

    imfs = matread("EnhData/Dataset_2/Signal/DecomposedIMFs_SEMD.mat")
    imf_AD_SEMD  = imfs["imf_AD"]
    imf_CR_SEMD  = imfs["imf_CR"]

    imfs = matread("EnhData/Dataset_2/Signal/DecomposedIMFs_CEMD.mat")
    imf_AD_CEMD  = imfs["imf_AD"]
    imf_CR_CEMD  = imfs["imf_CR"]
    try
        mkdir("EnhData/Dataset_2/SplitData/")
    catch
    end
    for i in 1:10
        train_ind_AD, test_ind_AD, train_ind_CR, test_ind_CR=generate_index(data_AD,data_CR,i)
        train_AD, test_AD=generate_realcols(data_AD, train_ind_AD, test_ind_AD)
        train_CR, test_CR=generate_realcols(data_CR, train_ind_CR, test_ind_CR)

        train_AD_enh=generate_artifact(imf_AD_MEMD,train_ind_AD,i,0);
        train_CR_enh=generate_artifact(imf_CR_MEMD,train_ind_CR,i,1);

    
        try
            mkdir("EnhData/Dataset_2/SplitData/MEMD")
        catch
        end
        file_memd="EnhData/Dataset_2/SplitData/MEMD/enhd_Data_$(i).mat"
        dict_memd=Dict("train_AD_enh" => train_AD_enh,"train_CR_enh" => train_CR_enh,
                        "train_AD"    => train_AD,     "train_CR"  => train_CR,
                        "test_AD"    => test_AD,       "test_CR"  => test_CR)
        matwrite(file_memd,dict_memd)
        

        train_AD_enh=generate_artifact(imf_AD_SEMD,train_ind_AD,i,0);
        train_CR_enh=generate_artifact(imf_CR_SEMD,train_ind_CR,i,1);
        try
            mkdir("EnhData/Dataset_2/SplitData/SEMD")
        catch
        end
        file_semd="EnhData/Dataset_2/SplitData/SEMD/enhd_Data_$(i).mat"
        dict_semd=Dict("train_AD_enh" => train_AD_enh,"train_CR_enh" => train_CR_enh,
                        "train_AD"    => train_AD,     "train_CR"  => train_CR,
                        "test_AD"    => test_AD,       "test_CR"  => test_CR)
        matwrite(file_semd,dict_semd)

        train_AD_enh=generate_artifact(imf_AD_CEMD,train_ind_AD,i,0);
        train_CR_enh=generate_artifact(imf_CR_CEMD,train_ind_CR,i,1);
        try
            mkdir("EnhData/Dataset_2/SplitData/CEMD")
        catch
        end
        file_cemd="EnhData/Dataset_2/SplitData/CEMD/enhd_Data_$(i).mat"
        dict_cemd=Dict("train_AD_enh" => train_AD_enh,"train_CR_enh" => train_CR_enh,
                        "train_AD"    => train_AD,     "train_CR"  => train_CR,
                        "test_AD"    => test_AD,       "test_CR"  => test_CR)
        matwrite(file_cemd,dict_cemd)
    end
end

generate_artifact_dataset1()
generate_artifact_dataset2()
