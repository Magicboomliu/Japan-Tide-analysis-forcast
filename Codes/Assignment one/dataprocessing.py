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

    utils.plot_image(simple_average_data,"Time(hours)","Tide Value","{} Tide reduced Data".format(city_name),"{} Data".format(city_name),saved="{}/{}_processed".format(saved_folder,city_name))

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




def process_in_question_two(datalist,city_name_list,saved_folder):
    simple_average_data_list = []
    R_value =[]
    for i, data in enumerate(datalist):
        simple_average_data = utils.SampleMovingAverage(data)
        simple_average_data = utils.remove_bias_and_trend(simple_average_data)
        simple_average_data_list.append(simple_average_data)


    R_1_2 = utils.compute_the_R(data1=simple_average_data_list[0],data2=simple_average_data_list[1])
    print("R value of 1 and 2 is : ",R_1_2)
    utils.draw_cross_corr(simple_average_data_list[0],simple_average_data_list[1],saved_cities=city_name_list[:2],saved_folder=saved_folder)
    R_1_3 = utils.compute_the_R(data1=simple_average_data_list[0],data2=simple_average_data_list[2])
    print("R value of 1 and 3 is : ",R_1_3)
    utils.draw_cross_corr(simple_average_data_list[0],simple_average_data_list[2],saved_cities=[city_name_list[0],city_name_list[2]],saved_folder=saved_folder)






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
    # arima_model = predict_in_ARIMA_model(full_data_list,idx=0,city_name="hatotade",saved_folder="hatakode")
    # further_five_days = utils.predict_arima(arima_model,5)
    # print("predict result: \n",further_five_days)


    # Draw the cross correlation Graph
    city_lists =["hakotade","otaru","osaka"]
    process_in_question_two(full_data_list,city_name_list=city_lists,saved_folder='figures')



    


    
    


    
