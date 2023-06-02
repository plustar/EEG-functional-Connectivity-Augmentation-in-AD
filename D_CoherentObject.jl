using MATLAB
using MAT
using BenchmarkTools
using LinearAlgebra
using Flux:unstack,stack
using Statistics
using ProgressBars
function coherent_matrix(X, W, N, findex, Fs)
    n_chn = size(X)[2]
    cotensor = zeros(Complex, n_chn, n_chn, Int(N / 2) + 1)
    for c = 1:n_chn
        cotensor[c,c,:] = mxcall(:cpsd, 1, X[:,c], X[:,c], W, [], N, Fs)
    end
    for c = 1:n_chn
        for h = (c + 1):n_chn
            cotensor[c,h,:] = mxcall(:cpsd, 1, X[:,c], X[:,h], W, [], N, Fs)
            cotensor[c,h,:] ./= sqrt.(cotensor[c,c,:] .* cotensor[h,h,:])
        end
    end
    cotensor = (abs2.(cotensor)) .* (-diagm(ones(n_chn)) .+ 1)
    cotensor = 0.5 * log.((cotensor .+ 1) ./ (-cotensor .+ 1))
    comatrix = mean(real.(cotensor[:,:,findex]), dims=3)[:,:,1]
    comatrix = (exp.(2 * comatrix) .- 1) ./ (exp.(2 * comatrix) .+ 1)
    comatrix = comatrix + comatrix' + diagm(ones(n_chn))
    return comatrix
end

function cm_d(X,electro_num,reference_electrodes,ind_group,W,N,findex,Fs)
    n_trl=size(X,1)
    set_matrix = zeros(electro_num - 2, electro_num - 2, n_trl)
    lbl_matrix = zeros(n_trl)
    x = mean(X[:,reference_electrodes,:], dims=2)
    X = X[:,setdiff(1:electro_num, reference_electrodes),:] .- x
    X = permutedims(X, [1,3,2])
    pbar=tqdm(1:n_trl)
    for i in pbar
        set_matrix[:,:,i] .= coherent_matrix(X[i,:,:], W, N, findex, Fs)
        lbl_matrix[i] = ind_group
    end
    return set_matrix, lbl_matrix
end
function cms_d(data,group,electro_num,reference_electrodes,W,N,findex,Fs)
    ind_group=1
    X = data["train_" * group[ind_group]]
    train_AD,     lbl_train_AD    =cm_d(X,electro_num,reference_electrodes,ind_group,W,N,findex,Fs)
    X = data["test_" * group[ind_group]]
    test_AD,      lbl_test_AD     =cm_d(X,electro_num,reference_electrodes,ind_group,W,N,findex,Fs)
    X = data["train_" * group[ind_group]*"_enh"]
    train_AD_enh, lbl_train_AD_enh=cm_d(X,electro_num,reference_electrodes,ind_group,W,N,findex,Fs)
    ind_group=2
    X = data["train_" * group[ind_group]]
    train_CR,     lbl_train_CR    =cm_d(X,electro_num,reference_electrodes,ind_group,W,N,findex,Fs)
    X = data["test_" * group[ind_group]]
    test_CR,      lbl_test_CR     =cm_d(X,electro_num,reference_electrodes,ind_group,W,N,findex,Fs)
    X = data["train_" * group[ind_group]*"_enh"]
    train_CR_enh, lbl_train_CR_enh=cm_d(X,electro_num,reference_electrodes,ind_group,W,N,findex,Fs)
    comatrix = Dict("train_AD" => train_AD,     "lbl_train_AD" => lbl_train_AD,
                "train_CR" => train_CR,         "lbl_train_CR" => lbl_train_CR,
                "test_AD" => test_AD,           "lbl_test_AD" => lbl_test_AD,
                "test_CR" => test_CR,           "lbl_test_CR" => lbl_test_CR,
                "train_AD_enh" => train_AD_enh, "lbl_train_AD_enh" => lbl_train_AD_enh,
                "train_CR_enh" => train_CR_enh, "lbl_train_CR_enh" => lbl_train_CR_enh,
                )
    return comatrix
end
function generate_enh_dadesplymouth(fband,randind,emd)
    N = min(2560 / 5, 2^10);
    W = mxcall(:hamming, 1, N);
    Fs = 128;
    f = (0:Int64(N / 2)) * Fs / N
    findex = (floor.(f) .>= fband[1]) .* (floor.(f) .<= fband[2])
    electro_num = 21
    reference_electrodes = [8, 14]
    group = ["AD","CR"]

    read_filename = "EnhData/Dataset_1/SplitData/"*emd*"/enhd_Data_$(randind).mat"
    data = matread(read_filename)

    comatrix=cms_d(data,group,electro_num,reference_electrodes,W,N,findex,Fs)
    try
        mkdir("EnhData/Dataset_1/CoMatrix/"*emd*"/")
    catch
    end
    matwrite("EnhData/Dataset_1/CoMatrix/"*emd*"/comatrix_$(fband[1])_$(fband[2])_enh_rind_$(randind).mat", comatrix)
    data=[]
    return
end
function cm_m(X,electro_num,ind_group,W,N,findex,Fs)
    n_trl=size(X,1)
    set_matrix = zeros(electro_num, electro_num, n_trl)
    lbl_matrix = zeros(n_trl)
    X = permutedims(X, [1,3,2])
    pbar=tqdm(1:n_trl)
    for i in pbar
        set_matrix[:,:,i] .= coherent_matrix(X[i,:,:], W, N, findex, Fs)
        lbl_matrix[i] = ind_group
    end
    return set_matrix, lbl_matrix
end
function cms_m(data,electro_num,group,W,N,findex,Fs)
    ind_group=1
    X = data["train_" * group[ind_group]]
    train_AD,     lbl_train_AD    =cm_m(X,electro_num,ind_group,W,N,findex,Fs)
    X = data["test_" * group[ind_group]]
    test_AD,      lbl_test_AD     =cm_m(X,electro_num,ind_group,W,N,findex,Fs)
    X = data["train_" * group[ind_group]*"_enh"]
    train_AD_enh, lbl_train_AD_enh=cm_m(X,electro_num,ind_group,W,N,findex,Fs)
    ind_group=2
    X = data["train_" * group[ind_group]]
    train_CR,     lbl_train_CR    =cm_m(X,electro_num,ind_group,W,N,findex,Fs)
    X = data["test_" * group[ind_group]]
    test_CR,      lbl_test_CR     =cm_m(X,electro_num,ind_group,W,N,findex,Fs)
    X = data["train_" * group[ind_group]*"_enh"]
    train_CR_enh, lbl_train_CR_enh=cm_m(X,electro_num,ind_group,W,N,findex,Fs)
    comatrix = Dict("train_AD" => train_AD,     "lbl_train_AD" => lbl_train_AD,
                "train_CR" => train_CR,         "lbl_train_CR" => lbl_train_CR,
                "test_AD" => test_AD,           "lbl_test_AD" => lbl_test_AD,
                "test_CR" => test_CR,           "lbl_test_CR" => lbl_test_CR,
                "train_AD_enh" => train_AD_enh, "lbl_train_AD_enh" => lbl_train_AD_enh,
                "train_CR_enh" => train_CR_enh, "lbl_train_CR_enh" => lbl_train_CR_enh,
                )
    return comatrix
end
function generate_enh_mushadatabase(fband,randind,emd)
    N = min(4000 / 5, 2^10);
    W = mxcall(:hamming, 1, N);
    Fs = 200;
    f = (0:Int64(N / 2)) * Fs / N
    findex = (floor.(f) .>= fband[1]) .* (floor.(f) .<= fband[2])
    group = ["AD", "CR"]
    group_item_num = [25, 47]
    electro_num=21
    read_filename = "EnhData/Dataset_2/SplitData/"*emd*"/enhd_Data_$(randind).mat"
    data = matread(read_filename)
    comatrix=cms_m(data,electro_num,group,W,N,findex,Fs)
    try
        mkdir("EnhData/Dataset_2/CoMatrix/"*emd*"/")
    catch
    end
    matwrite("EnhData/Dataset_2/CoMatrix/"*emd*"/comatrix_$(fband[1])_$(fband[2])_enh_rind_$(randind).mat", comatrix)
    data=[]
    return
end
fbands = unstack([4 8;8 10;10 13;13 30], 1)
try
    mkdir("EnhData/Dataset_1/CoMatrix/")
    mkdir("EnhData/Dataset_2/CoMatrix/")
    mkdir("EnhData/Dataset_1/CoMatrix/MEMD")
    mkdir("EnhData/Dataset_1/CoMatrix/SEMD")
    mkdir("EnhData/Dataset_1/CoMatrix/CEMD")
    mkdir("EnhData/Dataset_2/CoMatrix/MEMD")
    mkdir("EnhData/Dataset_2/CoMatrix/SEMD")
    mkdir("EnhData/Dataset_2/CoMatrix/CEMD")
catch
end
for emd in ["MEMD", "SEMD", "CEMD"]
    for i in 1:10
        for fband in fbands
            
            try
                read_filename = "EnhData/Dataset_1/CoMatrix/"*emd*"/comatrix_$(fband[1])_$(fband[2])_enh_rind_$(i).mat"
                data = matread(read_filename)
            catch
                println("dadesplymouth ", emd, i, fband)
                generate_enh_dadesplymouth(fband,i,emd)
            end

            try
                read_filename = "EnhData/Dataset_2/CoMatrix/"*emd*"/comatrix_$(fband[1])_$(fband[2])_enh_rind_$(i).mat"
                data = matread(read_filename)
            catch
                println("mushadatabase ", emd, i, fband)
                generate_enh_mushadatabase(fband,i,emd)
            end

            
        end
    end
end
# generate_dadesplymouth([4,30])
# generate_mushadatabase([4,30])]
