import numpy as np
import math
import json
import datetime
# import canopy
import multiprocessing as mp
from sklearn.cluster import KMeans
from sklearn import preprocessing
#from Data_norm import normalise_data

def read_dataset(file_url):
    '''
    读取文件中的数据集
    :param file_url: 数据集的地址
    :return: 数据集
    '''
    np.set_printoptions(suppress=True, linewidth=300)
    #设置浮点精度   linewidth（int每行用于插入的字符数# 换行符(默认为75)）
    #科学记数法启用 suppress=True用固定点打印浮点数符号，当前精度中的数字等于零将打印为零。
    data = np.loadtxt(file_url, dtype=float, delimiter=',')   #delimiter:加载文件分隔符 dtype：读取数据后数据的类型
    return data

def split_label(dataset):
    '''
    分割数据集，将数据集的属性和标签分隔开
    :param dataset: 获取的数据集
    :return: 标签和不带标签的数据集
    '''
    label = dataset[:, -1]    #取数据的最后一列
    data_without_label = np.delete(dataset,-1,axis=1)   #axis=1列向量
    return label,data_without_label

def zeros_one_normalization(dataset):
    min_max_scaler = preprocessing.MinMaxScaler()
    zero_one_dataset = min_max_scaler.fit_transform(dataset)
    return zero_one_dataset

# def compute_threshold(data):
#     data_mean=np.mean(data)
#     distance=np.zeros(data.shape)
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             distance[i][j]=math.sqrt((data[i][j]-data_mean)**2)
#     matrix_Max=distance.max()
#     matrix_Min=distance.min()
#     para1=matrix_Max-matrix_Min
#     para2=matrix_Max/2
#     return para1,para2


def K_means_complish(dataset,number):

    estimator = KMeans(n_clusters=number,random_state=1)  # 构造聚类器np.transpose
    estimator.fit(np.transpose(dataset))  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    # label_pred=[0, 0, 0, 1, 1, 1, 2, 2, 2]
    centers = estimator.cluster_centers_  # (np.transpose(zero_one_dataset))
    return label_pred,centers


def union_function(ori_data,cluster_result):
    class_info = set(cluster_result)
    res_dict = dict()
    feature_dict=dict()
    for i in class_info:
        res_dict[i]=np.zeros((ori_data.shape[0],1))
    for i in range(len(cluster_result)):
        res_dict[cluster_result[i]]=np.hstack((res_dict[cluster_result[i]],ori_data[:,i:i+1]))
        feature_dict[i]=cluster_result[i]
    for key in res_dict:
        res_dict[key]=res_dict[key][:,1:]
    #print(res_dict)
    return res_dict,feature_dict

def compute_class_var(cluster_dict,center):
    current_class_var=np.zeros((center.shape[0],1))

    for i in range(current_class_var.shape[0]):
        a=center[i,:]
        #print(a)
        current_class_var[i][0]=np.std(np.transpose(a),ddof=0)
    return current_class_var

def sample_similarity(x,y,sigma,yita,d_index,para2):
    '''
    样本内相似性的计算
    :param x: 样本的参数
    :param y: 样本的参数
    :param sigma: 样本的标准差
    :return: 样本内不同样本相似性
    '''
    # return (math.exp(-((x - y) ** 2) / (2 * (sigma *para2) ** 2)))
    return ((yita*math.exp(-((x-y)**2)/(2*(para2*sigma)**2)))+(1-yita)*1)**d_index

def disjust_ratio(corr_feature,corr_feature_dict,corr_class_var,para1):
        attribute_cluster_class = corr_feature_dict[corr_feature]
        attribute_cluster_var = corr_class_var[attribute_cluster_class][0]
        yita =  para1
        d_index = 1
        # if attribute_cluster_var >= 0.3:
        #     yita = 1 -para1
        #     d_index = 1
        # elif 0.2 <= attribute_cluster_var < 0.3:
        #     yita = 0.9
        #     d_index = 1
        # elif 0 <= attribute_cluster_var < 0.2:
        #     yita = 0.6
        #     # yita=1
        #     d_index = 1

        return yita,d_index
def compute_similarity1(name,data,corr_feature_dict,corr_class_var,feature_position,para1,para2):
    '''0
    计算样本相似性矩阵
    :param data: 同一属性下不同样本属性值（列向量）
    :return: 相似性矩阵
    '''
    res = dict()  # 建立一个字典
    sigma_sum = 0
    feature_num = data.shape[1]  # 计算特征的数目
    # print(feature_num)
    # min_value = 100000000
    # max_value = 0
    # for feature in range(feature_num):  # 计算每一个特征的相似度，每次都存放在一个矩阵中
    #     recent_sigma = np.std(data[:, feature],ddof=1)  # 计算在当前特征下的标准差
    #     if recent_sigma < min_value:
    #         min_value = recent_sigma
    #     if recent_sigma > max_value:
    #         max_value = recent_sigma
    # print(min_value, max_value)
    for feature in range(feature_num):
        recent_array = np.zeros((data.shape[0], data.shape[0]))
        recent_feature = data[:, feature]  # 当前需要计算的一列特征
        # print("123456789", max(recent_feature), min(recent_feature))
        recent_sigma = np.std(data[:, feature], ddof=1)
        # print(recent_sigma)
        recent_sigma_yita = 1 - recent_sigma * 0.7
        # if recent_sigma * 4.0 < 1:
        #     recent_sigma_yita = recent_sigma * 4.0
        # else:
        #     recent_sigma_yita = 1
        (yita, index) = disjust_ratio(feature_position[feature], corr_feature_dict, corr_class_var,recent_sigma_yita)
        # print(recent_sigma, recent_sigma_yita)
        for x_position in range(data.shape[0]):
            for y_position in range(data.shape[0]):
                recent_array[x_position][y_position] = sample_similarity(recent_feature[x_position], recent_feature[y_position], recent_sigma, yita,index,para2)
        res[feature] = recent_array   #将计算后的相似度存在于一个字典中，并用属性进行标注 （属性+相似度）
    return {name:res}

def fuzzy_implicator(x,y):
    '''
    计算fuzzy_implicator
    :param x: 参数
    :param y: 参数
    :return: 计算值
    '''
    return min(1-x+y,1)

def equal_judge(x,y):
    '''
    判断样本的标签与类别标签是否一致
    :param x: 样本的标签
    :param y: 类别的标签
    :return: 一致返回1，不一致返回0
    '''
    if x==y:
        return 1
    else:
        return 0

def lower_approximate(similarity_arr,local_label):
    '''
    先计算样本在不同标签下同一属性下近似，然后得到属性正域
    :param similarity_arr: 不同属性下的相似性矩阵
    :param local_label: 样本的标签
    :return: 不同属性下的正域
    '''
    tmp = dict()  #建立一个字典
    label_info = set(list(local_label))   #将标签转换为一个列表，然后存放在一个集合中
    for class_info in label_info:   #下近似矩阵的计算
        recent_arr = np.zeros(similarity_arr.shape)
        for sample_x in range(similarity_arr.shape[0]):
            for sample_y in range(similarity_arr.shape[0]):
                judge=equal_judge(local_label[sample_y],class_info)   #判断当前样本所属的类别
                recent_arr[sample_x][sample_y]=fuzzy_implicator(similarity_arr[sample_x][sample_y],judge)
        tmp[class_info] = recent_arr   #将计算的下近似矩阵存放在字典中，并用属性类别进行标记   属性类别+下近似矩阵

    res = dict()   #建立一个字典
    for key in tmp:   #对字典中的关键字进行循环
        tmp_list = list()   #建立一个列表
        for i in range(tmp[key].shape[0]):  #关键字后接下近似矩阵的行数
            tmp_list.append(min(tmp[key][i,:]))   #将下近似每行的最小值添加到列表中
        res[key]=tmp_list   #将取自每行的最小值放到一个字典中，前面用属性标签进行标记

    positive = [0 for i in range(local_label.shape[0])] #建立一个全为0的行向量，用来存放后面将要取得最大值
    for key in res:
        for position in range(len(positive)):
            positive[position] = max(positive[position],res[key][position])   #取最大值
    return positive

def single_feature_positive(local_similatity_dict,recent_label):
    '''
    计算单个特征的正域
    :param local_similatity_dict: 属性相似度的字典
    :return: 样本在不同标签集合下的下近似值
    '''
    res = dict()
    for key in local_similatity_dict:    #遍历字典中的关键字key
        res[key]=lower_approximate(local_similatity_dict[key], recent_label)
    return res

def single_feature_positive1(name,local_similatity_dict,recent_label):
    '''
    计算单个特征的正域
    :param local_similatity_dict: 属性相似度的字典
    :return: 样本在不同标签集合下的下近似值
    '''
    res = dict()
    for key in local_similatity_dict:    #遍历字典中的关键字key
        res[key]=lower_approximate(local_similatity_dict[key], recent_label)
    return {name:res}

def feature_dependency(local_positive_dict):
    '''
    属性依赖度值的计算
    :param local_positive_dict:属性在不同标签集合下的正域值
    :return:依赖度值
    '''
    res=dict()
    for key in local_positive_dict:
        res[key]=sum(local_positive_dict[key])/len(local_positive_dict[key])
    return res

def feature_dependency1(name,local_positive_dict):
    '''
    属性依赖度值的计算
    :param local_positive_dict:属性在不同标签集合下的正域值
    :return:依赖度值
    '''
    res=dict()
    for key in local_positive_dict:
        res[key]=sum(local_positive_dict[key])/len(local_positive_dict[key])
    return {name:res}


def max_feature_dependency(local_single_attribute_dependency):
    '''
    计算最大的单属性依赖度
    :param local_single_attribute_dependency:单依赖度值
    :return: 最大单属性
    '''
    return max(local_single_attribute_dependency.keys(), key=(lambda k: local_single_attribute_dependency[k]))
#返回最大单属性依赖度及对应的属性

def t_norm(x,y):
    return max(x+y-1, 0)
# def t_norm(x,y):
#     return min(x,y)

def combine_compute_similarity(local_recent_similarity,local_best_similarity):
    '''
    组合属性相似度的度量
    :param local_recent_similarity:  当前与最有属性相组合属性的相似性矩阵
    :param local_best_similarity: 已经计算所得最有属性的相似性矩阵
    :return:  组合属性的相似性矩阵
    '''
    res_arr = np.zeros(local_recent_similarity.shape)
    for position_x in range(local_recent_similarity.shape[0]):
        for position_y in range(local_recent_similarity.shape[0]):
            res_arr[position_x][position_y]= t_norm(local_recent_similarity[position_x][position_y],local_best_similarity[position_x][position_y])
    return res_arr

def combine_compute_similarity(local_recent_similarity,local_best_similarity):
    '''
    组合属性相似度的度量
    :param local_recent_similarity:  当前与最有属性相组合属性的相似性矩阵
    :param local_best_similarity: 已经计算所得最有属性的相似性矩阵
    :return:  组合属性的相似性矩阵
    '''
    res_arr = np.zeros(local_recent_similarity.shape)
    for position_x in range(local_recent_similarity.shape[0]):
        for position_y in range(local_recent_similarity.shape[0]):
            res_arr[position_x][position_y]= t_norm(local_recent_similarity[position_x][position_y],local_best_similarity[position_x][position_y])
    return res_arr

def combine_compute_similarity1(name,local_recent_similarity_dict,local_best_similarity):
    '''
    组合属性相似度的度量
    :param local_recent_similarity:  当前与最有属性相组合属性的相似性矩阵
    :param local_best_similarity: 已经计算所得最有属性的相似性矩阵
    :return:  组合属性的相似性矩阵
    '''
    if len(local_recent_similarity_dict) == 0:
        return {name:None}
    res = []
    for key in local_recent_similarity_dict:
        local_recent_similarity = local_recent_similarity_dict[key]
        res_arr = np.zeros(local_recent_similarity.shape)
        for position_x in range(local_recent_similarity.shape[0]):
            for position_y in range(local_recent_similarity.shape[0]):
                res_arr[position_x][position_y]= t_norm(local_recent_similarity[position_x][position_y],local_best_similarity[position_x][position_y])
        res.append((key,res_arr))
    return {name:res}


def cutDictFunc(dict_info,core_num):
    res = []
    nums = len(dict_info)//core_num
    tmp_dict = dict()
    for key in dict_info:
        if len(tmp_dict)>nums:
            res.append(tmp_dict)
            tmp_dict = dict()
        tmp_dict[key]=dict_info[key]
    res.append(tmp_dict)
    return (res)

if __name__ == "__main__":
    start_t = datetime.datetime.now()
    num_cores = int(mp.cpu_count()) - 1
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores)
    # with open("D:/Python/experience/cluster.json", "r", encoding="utf-8") as f:
    #     cluster_number_dict = json.loads(f.readline())
    # print(cluster_number_dict)
    modified_list = [5]
    modified_sigma_list = [1]
    # var_data = {"sonar.csv": [0.005038978502417537, 0.2641126744548476],
    #             "colon.csv": [16.039575355559666, 4058.875022485207],
    #             "ALLAML.csv": [26.70311109983284, 15893.556424582908],
    #             "GLIOMA.csv": [0.11224311166528232, 0.7646973803424373],
    #             "LSVT.csv": [1.585401330959145e-09, 14136057419.642458]}
    for k in range(1,2):
        # "colon.csv", "ALLAML.csv", "GLIOMA.csv",
        # "sonar.csv", "wdbc.csv", "dermatology.csv", "wpbc.csv", "DLBCL.csv", "colon.csv",
        # "ALLAML.csv", "GLIOMA.csv", "lung.csv", "Lymphoma.csv", "LSVT.csv", "Leukemia.csv",
        # "setapProcessT1.csv", "arcene.csv"
        for data_name in ["wine.csv"]:
            for modified_yita in modified_list:
                for modified_sigma in modified_sigma_list:
                    ori_data = (read_dataset("D:/Python/DATA/new/{}".format(data_name)))
                    ori_data_var = [1, 1]
                    f = open("./example_3.txt", "a", encoding="utf-8")
                    remind_feature_set = [i for i in range(ori_data.shape[1]-1)]    #剩余的属性集合
                    ori_dependency = 0    #初始依赖度
                    label,data_no_label = split_label(ori_data)   #得到标签和没有标签的数据
                    attribute_number = data_no_label.shape[1]
                    zero_to_one_data_no_label = zeros_one_normalization(data_no_label)
                    print(zero_to_one_data_no_label)    #打印归一化后的特征
                    [cluster_result, cluster_centers] = K_means_complish(zero_to_one_data_no_label,k)
                    same_class_cluster,feature_dict_info = union_function(zero_to_one_data_no_label, cluster_result)
                    # print(feature_dict_info)
                    class_var = compute_class_var(same_class_cluster, cluster_centers)
                    # print('00',class_var)
                    average_index = int((data_no_label.shape[1]/num_cores)+1)
                    data_no_label_dict = {}
                    for i in range(num_cores):
                        data_no_label_dict["task{}".format(i + 1)] = [zero_to_one_data_no_label[:, average_index * i:average_index * (i + 1)],feature_dict_info, class_var, [i for i in range(average_index * i, (i + 1) * average_index)]]
                    print("hello")
                    results = [pool.apply_async(compute_similarity1, args=(name,value_information[0],value_information[1],
                                            value_information[2],value_information[3],ori_data_var,modified_sigma))for name, value_information in data_no_label_dict.items()]
                    # print(results.get())
                    results_1 = [p.get() for p in results]

                    end_t = datetime.datetime.now()
                    elapsed_sec = (end_t - start_t).total_seconds()
                    print("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")

                    start_t1 = datetime.datetime.now()

                    similarity_dict = dict()
                    for task in results_1:
                        for key in task:
                            times = int(key[4:])-1
                            for s in task[key]:
                                similarity_dict[average_index*times+s]=task[key][s]
                    print(similarity_dict)
                    end_t1 = datetime.datetime.now()
                    elapsed_sec = (end_t1 - start_t1).total_seconds()
                    print("合并计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")
                    similarity_matrix = []
                    start_t2 = datetime.datetime.now()
                    best_feature_subset = []    #最优特征子集集合
                    cutted_similarity_dict_list = cutDictFunc(similarity_dict,num_cores)
                    similarity_matrix_dict = {}
                    for i in range(len(cutted_similarity_dict_list)):
                        similarity_matrix_dict["task{}".format(str(i+1))] =[cutted_similarity_dict_list[i]]

                    feature_positive_region_results = [pool.apply_async(single_feature_positive1,args=(name, value_information[0], label)) for name, value_information in similarity_matrix_dict.items()]

                    feature_positive_region_results_1 = [p.get() for p in feature_positive_region_results]
                    feature_positive_region = {}

                    for i in feature_positive_region_results_1:
                        for task in i:
                            for key in i[task]:
                                feature_positive_region[key] = i[task][key]

                    cutted_positive_dict_list = cutDictFunc(feature_positive_region, num_cores)
                    feature_positive_region_dict = dict()
                    for i in range(len(cutted_positive_dict_list)):
                        feature_positive_region_dict["task{}".format(str(i + 1))] = [cutted_positive_dict_list[i]]
                    feature_dependency_results = [pool.apply_async(feature_dependency1, args=(name, value_information[0])) for name, value_information in feature_positive_region_dict.items()]

                    feature_dependency_results_1 = [p.get() for p in feature_dependency_results]
                    dependency_dict = dict()
                    for i in feature_dependency_results_1:
                        for task in i:
                            for key in i[task]:
                                dependency_dict[key] = i[task][key]

                    max_dependency_feature=max_feature_dependency(dependency_dict)
                    best_feature_subset.append(max_dependency_feature)
                    remind_feature_set.remove(max_dependency_feature)
                    recent_similarity_arr = similarity_dict[max_dependency_feature]
                    # # print(dependency_dict)
                    print("当前选择特征{},依赖度为{}".format(max_dependency_feature+1,dependency_dict))
                    # f.write("当前选择特征{},依赖度为{}".format(max_dependency_feature, dependency_dict))
                    # f.write("\n")
                    end_t2 = datetime.datetime.now()
                    elapsed_sec = (end_t2 - start_t).total_seconds()
                    print("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")
                    #
                    if dependency_dict[max_dependency_feature] == 1 or not remind_feature_set or dependency_dict[max_dependency_feature]== 1:
                        print(best_feature_subset)
                        f.write("当前数据集:{}".format(data_name))
                        f.write("\n")
                        f.write("当前参数:{}".format(modified_sigma))
                        f.write("\n")
                        f.write("最终结果为：{}".format(best_feature_subset))
                        f.write("\n")
                    else:
                        while True:
                            combine_dict = dict()
                            tmp_list = []
                            for feature in remind_feature_set:
                                tmp_list.append(feature)
                            pool_list = []
                            for i in range(num_cores):
                                if i>=len(tmp_list):
                                    process_list = []
                                else:
                                    process_list = tmp_list[i*(int(len(tmp_list)/num_cores)+1):(i+1)*(int(len(tmp_list)/num_cores)+1)]
                                    # print(process_list)
                                process_dict = dict()
                                for feature in process_list:
                                    process_dict[feature]=similarity_dict[feature]
                                pool_list.append(process_dict)
                            pool = mp.Pool(num_cores)
                            combine_data_no_label_dict = {}
                            for i in range((num_cores)):
                                combine_data_no_label_dict['task{}'.format(i + 1)] = [pool_list[i], recent_similarity_arr]
                            combine_results = [pool.apply_async(combine_compute_similarity1, args=(name, combine_value_information[0], combine_value_information[1])) for name, combine_value_information in combine_data_no_label_dict.items()]
                            # print(combine_results)
                            combine_results_1 = [p.get() for p in combine_results]

                            for task_info in combine_results_1:
                                for key in task_info:
                                    if task_info[key]:
                                        for i in task_info[key]:
                                            combine_dict[i[0]] = i[1]

                            cutted_positive_dict_list_repeat = cutDictFunc(combine_dict,num_cores)
                            positive_dict = {}
                            for i in range(len(cutted_positive_dict_list_repeat)):
                                positive_dict["task{}".format(str(i + 1))] = [cutted_positive_dict_list_repeat[i]]
                            feature_positive_region_dicc = [pool.apply_async(single_feature_positive1,args=(name, combine_value_information[0],label))for name, combine_value_information in positive_dict.items()]

                            feature_positive_region_dicc_1 = [p.get() for p in feature_positive_region_dicc]

                            feature_positive_region = dict()
                            for i in feature_positive_region_dicc_1:
                                for task in i:
                                    for key in i[task]:
                                        feature_positive_region[key] = i[task][key]

                            down_dependecy_cuted_func = cutDictFunc(feature_positive_region, num_cores)
                            cuted_down_dependecy_dict = {}
                            for i in range(len(down_dependecy_cuted_func)):
                                cuted_down_dependecy_dict["task{}".format(str(i + 1))] = [down_dependecy_cuted_func[i]]


                            down_dependecy_results = [pool.apply_async(feature_dependency1, args=(name, combine_value_information[0]))for name, combine_value_information in cuted_down_dependecy_dict.items()]
                    #         # print(combine_results)
                            down_dependecy_results_1 = [p.get() for p in down_dependecy_results]
                            # print(down_dependecy_results_1)
                            dependency_dict=dict()
                            for i in down_dependecy_results_1:
                                for task in i:
                                    for key in i[task]:
                                        dependency_dict[key] = i[task][key]

                            max_dependency_feature = max_feature_dependency(dependency_dict)
                            if dependency_dict[max_dependency_feature] == 1 or len(remind_feature_set)==1:
                                best_feature_subset.append(max_dependency_feature)
                                print("当前选择特征{},依赖度为{}".format(max_dependency_feature+1, dependency_dict))
                                # f.write("当前选择特征{},依赖度为{}".format(max_dependency_feature, dependency_dict))
                                # f.write("\n")
                                print(best_feature_subset)
                                f.write("当前数据集:{}".format(data_name))
                                f.write("\n")
                                f.write("当前参数:{}".format(modified_sigma))
                                f.write("\n")
                                f.write("最终结果为：{}".format(best_feature_subset))
                                f.write("\n")
                                end_t3 = datetime.datetime.now()
                                elapsed_sec = (end_t3 - start_t).total_seconds()
                                # print("特征选择计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")
                                break
                            elif ori_dependency == dependency_dict[max_dependency_feature]:
                                print(best_feature_subset)
                                f.write("当前数据集:{}".format(data_name))
                                f.write("\n")
                                f.write("当前参数:{}".format(modified_sigma))
                                f.write("\n")
                                f.write("最终结果为：{}".format(best_feature_subset))
                                f.write("\n")
                                break
                            else:
                                best_feature_subset.append(max_dependency_feature)
                                ori_dependency = dependency_dict[max_dependency_feature]
                                remind_feature_set.remove(max_dependency_feature)
                                recent_similarity_arr = combine_dict[max_dependency_feature]
                                print("当前选择特征{},依赖度为{}".format(max_dependency_feature+1, dependency_dict))
                                # f.write("当前选择特征{},依赖度为{}".format(max_dependency_feature, dependency_dict))
                                f.write("\n")
                                end_t4 = datetime.datetime.now()
                                elapsed_sec = (end_t4 - start_t).total_seconds()
                                # print("特征选择计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")
                    f.close()

