using MAT
using Random
using DataFrames

function readmat_DadesPlymouth(group, index)
    file_index = group * "$(index)"
    read_filename = "Data/DadesPlymouth/Signal/Plymouth_AssA_" * file_index * "_clean.mat"
    data = matread(read_filename)
    return data["X"]
end
function read_DadesPlymouth()
    group = ["AD", "Ctr"]
    group_item_num = [17, 24]
    data_AD = zeros(17, 21, 2560)
    data_CR = zeros(24, 21, 2560)
    for ind_group_item in 1:group_item_num[1]
        data_AD[ind_group_item,:,:] = readmat_DadesPlymouth("AD", ind_group_item)[:,1:2560]
    end
    for ind_group_item in 1:group_item_num[2]
        data_CR[ind_group_item,:,:] = readmat_DadesPlymouth("Ctr", ind_group_item)[:,1:2560]
    end
    return data_AD, data_CR
end
function readmat_MushaDatabase(group, index)
    file_index = group * "$(index)"
    if index < 10
        read_filename = "Data/MushaDatabase/Signal/" * group * "_20s_00$(index).mat"
    else
        read_filename = "Data/MushaDatabase/Signal/" * group * "_20s_0$(index).mat"
    end
    data = matread(read_filename)
    return data["x"]
end
function read_MushaDatabase()
    group = ["MildAD", "Normal"]
    group_item_num = [25, 47]
    group_arifact = [7 18 21 missing missing missing missing missing missing;2 7 9 14 17 22 26 31 43]
    electro_num = 21
    
    Ind1=setdiff(1:group_item_num[1], group_arifact[1,:])
    data_MildAD = zeros(length(Ind1), 21, 4000)
    
    for ind_group_item in 1:length(Ind1)
        data_MildAD[ind_group_item,:,:] = readmat_MushaDatabase("MildAD", Ind1[ind_group_item])
    end
    Ind2=setdiff(1:group_item_num[2], group_arifact[2,:])
    data_Normal = zeros(length(Ind2), 21, 4000)
    for ind_group_item in 1:length(Ind2)
        data_Normal[ind_group_item,:,:] = readmat_MushaDatabase("Normal", Ind2[ind_group_item])
    end
    return data_MildAD, data_Normal
end

data_AD, data_CR = read_DadesPlymouth();
data=Dict("data_AD"=>data_AD, "data_CR"=>data_CR);
write_filename = "Data/DadesPlymouth/Signal/CombinedDataDade.mat";
matwrite(write_filename, data)

data_AD, data_CR = read_MushaDatabase();
data=Dict("data_AD"=>data_AD, "data_CR"=>data_CR);
write_filename = "Data/MushaDatabase/Signal/CombinedDataMush.mat";
matwrite(write_filename, data)