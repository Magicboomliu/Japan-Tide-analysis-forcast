from numpy import random
from numpy.core.numeric import full
from numpy.lib.function_base import percentile
import utils
import numpy as np
import matplotlib.pyplot as plt
# 画出某一天的数据
# 画出某一个时刻的数据
# 画出每天潮汐峰值的数据
def process_in_question_one(datalist,idx,city_name,ma_mode =0,saved_folder =None,saved=True):
    #First to do moving average
    simple_average_data = utils.SampleMovingAverage(datalist[idx])
    culmative_average_data = utils.CumulativeMovingAverage(datalist[idx])
    expontial_average_data = utils.ExponentialMovingAverage(datalist[idx])
    draw_data_list =[datalist[idx],simple_average_data,culmative_average_data,expontial_average_data]
    data_lengend =["Orginal {} Data".format(city_name),"Simple Moving average data","Culmulative Averge Data","Exponential Moving Average Data"]
    ma_data_list = [simple_average_data,culmative_average_data,expontial_average_data]
    # Draw the data
    utils.plot_all(draw_data_list,data_lengend,"Date(only show the point of 0:00 in the X axe)","Tide value ","Tide value per day of {} with Moving Average".format(city_name),xticks=xticks_para,figsize=(16,10),saved="{}/Moving_Average_{}.png".format(saved_folder,city_name))

    # Select to use which kind of Move Average Data, default is the simple one
    simple_average_data = ma_data_list[ma_mode]

    # Remove the Bias and Trend of the Data for further processing
    simple_average_data = utils.remove_bias_and_trend(simple_average_data)

    # Draw the ACF plot of the data
    utils.draw_acf(simple_average_data, saved="{}/{}_acf.png".format(saved_folder,city_name))

    # Draw the PACF plot of the data
    utils.draw_pacf(simple_average_data, saved='{}/{}_pacf.png'.format(saved_folder,city_name))

    # Draw the ampltitude Specturm and Power Specturm
    utils.draw_spectrum(simple_average_data,city_name,saved_folder)

    return  simple_average_data


def predict_in_ARIMA_model(datalist,idx,city_name,saved_folder):

    simple_average_data = utils.SampleMovingAverage(datalist[idx])
    # 平稳性检测
    is_stationarity = utils.stationarity_test(simple_average_data)
    # 如果不是平稳序列，进行差分操作
    num_diff =0
    while is_stationarity ==False:
        num_diff = num_diff +1
        simple_average_data=utils.one_order_diff(simple_average_data,content="{} oder difference".format(num_diff),saved="{}_{}_{}_diff_data".format(saved_folder,city_name,num_diff))
        is_stationarity = utils.stationarity_test(simple_average_data)

    p,q = utils.get_p_and_q(simple_average_data)
    print("p is {},  q is {} ".format(p,q))
    model = utils.buildARIMA(simple_average_data, p, q, num_diff)
    return  model









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

    # Draw All The Plots
    #utils.plot_all(full_data_list,data_lengends,"Date(only show the point of 0:00 in the X axe)","Tide value ","Tide value per day of Hakodate,Otaru and Osaka",xticks=xticks_para,figsize=(16,10),saved="figures/Original_data_all.png")


    # Use Hakodate as the analysis target
    #simple_average_data = process_in_question_one(full_data_list,idx=0,city_name="Hakotade",ma_mode=0,saved_folder="hakotade",saved=True)

    # Predict in ARIMA model
    arima_model = predict_in_ARIMA_model(full_data_list,idx=0,city_name="hatotade",saved_folder="hatakode")
    further_five_days = utils.predict_arima(arima_model,5)
    print("predict result: \n",further_five_days)


    
    










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

    
    


    
