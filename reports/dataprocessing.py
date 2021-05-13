from numpy import random
from numpy.lib.function_base import percentile
import utils
import numpy as np
import matplotlib.pyplot as plt
# 画出某一天的数据
# 画出某一个时刻的数据
# 画出每天潮汐峰值的数据

if __name__=="__main__":
    data_dir_list = ['data/Hakodate.txt','data/Otaru.txt','data/Osaka.txt']
    full_data_list =[]
    data_lengends=["Tide value per day of Hakodate","Tide value per day of Otaru","Tide value per day of Osaka"]
    for data in data_dir_list:
        dataframe = utils.pandas_dataframe_format(data)
        data_list = utils.merge_all_data(dataframe)
        full_data_list.append(data_list)
    
    xticks_list =[]
    for i in range(32):
        xticks_list.append("3-{}".format(i))
    xticks_para =[31*24,24,xticks_list]
    utils.plot_all(full_data_list,data_lengends,"Date(only show the point of 0:00 in the X axe)","Tide value ","Tide value per day of Hakodate,Otaru and Osaka",xticks=xticks_para,figsize=(16,10),saved="figures/Original_data_all.png")

   # utils.plot_image(full_data_list,"Data(only show the point of 0:00 in the x axe)","Tide value "," Tide value per day of Hakodate",' Tide Value observed Hakodate','r-',xticks=xticks_para)
   
    










    # max_data_list  = utils.use_max_each_rows(dataframe=dataframe)
    # utils.plot_image(max_data_list,"data","tide value","max tide value per day",'max tide observed','r--')
    # utils.draw_acf(max_data_list)
    # utils.draw_pacf(max_data_list)
    # utils.stationarity_test(max_data_list)
    # max_data_diff_one = utils.one_order_diff(max_data_list,"one order difference of the max tide")
    # utils.stationarity_test(max_data_diff_one)
    # p,q = utils.get_p_and_q(max_data_diff_one)
    # print("p is {},  q is {} ".format(p,q))
    # model = utils.buildARIMA(max_data_diff_one,p,q,1)
    # result  = utils.predict_arima(model,2)
    # print(result)
    # d = utils.trainModelWithARIMA(data= max_data_list)
    # sma_data = utils.ExponentialMovingAverage(max_data_list)
    # utils.plot_image(sma_data,"date","moving averged data","moving averged data","moving averaged data")
    # utils.calcuate_amplitude_spectrum(max_data_list,mode='ffshift',normalization=True)
    # ceps = utils.calculate_cepstrum(data=max_data_list,num_fft=1024)
    # utils.plot_image(ceps,"ceps","power","cep spectrum",'cep spectrum',saved="data/cep.png")
    # max_data_list2 =[]
    # for i in max_data_list:
    #     new = i + random.rand(1)
    #     max_data_list2.append(new[0])

    # r = utils.compute_the_R(data1=max_data_list,data2=max_data_list2)
    # print(r)

    
    


    
