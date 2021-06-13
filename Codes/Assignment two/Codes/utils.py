import pandas as pd
import  numpy as np

def exact_data(pandas_data):
    exact_time_list =[]
    exact_temperture_list=[]
    index_amount = len(pandas_data)
    for i in range(index_amount):
        exact_time = pandas_data.iloc[i][0]
        exact_temperture =pandas_data.iloc[i][1]
        exact_time_list.append(exact_time)
        exact_temperture_list.append(exact_temperture)
    return  np.array(exact_time_list), np.array(exact_temperture_list)

def read_csv_data(filename,hypyerparamer=6):
    '''
    :param filename: Input data file path
    :param hypyerparamer: the header data length, It depends
    :return: the temperature list(per hour), the exact time(per hour), the length of the data
    '''
    data = pd.read_csv(filename, header=None)
    value_data= data.iloc[hypyerparamer:]
    exact_time_list, exact_temp_list = exact_data(value_data)
    exact_temp_list= exact_temp_list.astype(np.float32)
    assert exact_time_list.shape == exact_temp_list.shape
    nans = np.isnan(exact_temp_list)

    # Nan value check and interplot
    for i,n in enumerate(nans):
        if n==True:
            values_front = exact_temp_list[i-4:i-1]
            values_back = exact_temp_list[i+1:i+4]
            exact_temp_list[i] = (sum(values_front)+sum(value_data))/6.0*1.0
            nans_index=i
    print("LOG: Successfully load the data")
    return exact_temp_list,exact_time_list,len(exact_temp_list)



