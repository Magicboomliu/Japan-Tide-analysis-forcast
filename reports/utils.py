from datetime import date
import  os
from numpy.lib import utils
import  pandas as pd
from pandas.core import api, series
import matplotlib.pyplot as plt
from pandas.core.algorithms import mode
from pandas.tseries.offsets import Day

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.fftpack import fft,fftshift,ifft
from scipy.fftpack import fftfreq
from scipy import signal
import numpy as np
import matplotlib.ticker as mtick
from scipy.signal import detrend

def merge_all_data(dataframe):
    cols,rows = dataframe.shape
    data_list=[]
    for i in range(cols):
        for j in range(rows):
            data_list.append(dataframe.iloc[i,j])
    return data_list

def plot_all(data_list,datalables,x_label,y_label,title,saved= None,figsize=(16,10),xticks= None):
    plt.figure(figsize=figsize)
    color =['r-','g-','b-','y-','p-']
    for i,d in enumerate(data_list):
        plt.plot(d,color[i],label= datalables[i])
    if xticks is not None:
        ticks_size = xticks[0]
        inter_ = xticks[1]
        xticks_content = xticks[2]
        ticks_list = list(range(0,ticks_size+inter_,inter_))
        plt.xticks(ticks_list,xticks_content)
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc = 1)
    if saved is not None:
        plt.savefig(saved)
    plt.show()



def get_max(list):
    max_val=-1e10
    max_index =0

    for i,value in enumerate(list):
        if value>max_val:
            max_val=value
            max_index = i
    
    return(max_index,max_val)

def series_to_list(series):
    return list(series)

def list_to_series(list):
    return pd.Series(list)

def read_txt_lines(filepath):
    with open(filepath, "r")  as f1:
        lines = f1.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def pandas_dataframe_format(data_dir):
    month_data=[]
    lines = read_txt_lines(data_dir)
    for i, line in enumerate(lines):
        splits = line.split()        
        splits = [int(i) for i in splits]
        data_per_day = dict()
        for j in range(1,25):
            data_per_day[j] = splits[j-1]
        month_data.append(data_per_day)
    df = pd.DataFrame(month_data)
    return  df

def use_max_each_rows(dataframe):
    cols,rows = dataframe.shape
    max_data_list=[]
    for i in range(cols):
        data_col =   series_to_list(dataframe.iloc[i])
        max_data = get_max(data_col)[1]
        max_data_list .append(max_data)
    return max_data_list

# 画图
def plot_image(x_data,x_label,y_label,title, x_data_label='x data',format_string='b-',saved= None,figsize=(16,6),xticks= None):
    plt.figure(figsize=figsize)
    plt.plot(x_data,format_string,label=x_data_label)
    if xticks is not None:
        ticks_size = xticks[0]
        inter_ = xticks[1]
        xticks_content = xticks[2]
        ticks_list = list(range(0,ticks_size+inter_,inter_))
        plt.xticks(ticks_list,xticks_content)
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    if saved is not None:
        plt.savefig(saved)
    plt.show()

# 自相关图
def draw_acf(data,saved=None,figsize=(16,10)):
    data =list_to_series(data)
    data =data.dropna()
    plot_acf(data)
    if saved is not None:
        plt.savefig(saved)
    plt.show()

# 偏自相关图
def draw_pacf(data,saved=None,figsize=(16,10)):
    data = list_to_series(data)
    data = data.dropna()
    plot_pacf(data)
    if saved is not None:
        plt.savefig(saved)
    plt.show()

# 返回AD
def ADF_Check(data):
    if isinstance(data,list):
        data = list_to_series(data)
        data = data.dropna()
    return ADF(data)

# 检查数据平稳性
def stationarity_test(data):
    adf = ADF_Check(data)
    state = False
    if (adf[1]>0.005):
        print( "This data is not a Stationary data, root is {}(>0.005), Need to Move average or diff operations ".format(adf[1]))
        state = False
    else:
        print("This is a Stationary data, root is {}(<0.005), you can use AR , MA, or ARMA ".format(adf[1]))
        state = True
    return state

# 数据求1阶差分
def one_order_diff(data,content=None,saved=None):
    data = list_to_series(data)
    data = data.diff().dropna()
    data.plot(label=content)
    plt.legend()
    plt.title(content)
    plt.savefig(saved)
    plt.show()
    return data

# 判断数据是否为白噪声
def white_noise_cal(data):
    if isinstance(data,list):
        data = list_to_series(data)
        data = data.dropna()
    acorr = acorr_ljungbox(data,lags=2)
    if acorr[1].mean()<0.05:
        print("Ok, it is not white noise,")
    else:
        print("The data is probably a Random Noise Data!")
    return acorr

# 求bic矩阵
def get_bic_matrix(data):
    if isinstance(data,list):
        data = list_to_series(data)
        data = data.dropna()
    data = data.astype('float')
    p_max = int(len(data)//500)
    q_max = int(len(data)//500)
    bic_matrix =[]
    
    for p in range(p_max+1):
        temp=[]
        for q in range(q_max+1):
            try:
                temp.append(ARIMA(data,(p,1,q)).fit().bic)
            except:
                temp.append(None)
        bic_matrix.append(temp)
    
    bic_matrix = pd.DataFrame(bic_matrix)
    return bic_matrix

# 根据bic矩阵求p和c
def get_p_and_q(data):
    bic_matrix= get_bic_matrix(data)
    p,q = bic_matrix.stack().idxmin()
    return(p,q)

# 构建ARIMA模型，根据p,d,q
def buildARIMA(data,p,q,d):
    model = ARIMA(data,(p,d,q)).fit()
    print(model.summary())
    return model

# 根据构建的模型进行预测和分析
def predict_arima(model,len):
    return model.forecast(len)

# 使用ARIMA进行时间序列预测
def trainModelWithARIMA(data):
    if isinstance(data,list):
        data = list_to_series(data)
        data =data.dropna()
        d = 0
        # 平滑性质检验
        stationarity  =stationarity_test(data)
        while not stationarity:
            data = one_order_diff(data)
            stationarity = stationarity_test(data)
            d = d +1
        #白噪声检验
        white = white_noise_cal(data)
        if white:
            return
        else:
            p,q = get_bic_matrix(data)
            model = buildARIMA(data,p,q,d)
            return model

# 简单平均平滑
def SampleMovingAverage(data):
    if isinstance(data,list):
        data = list_to_series(data)
        data = data.dropna()
    sma_data = data.rolling(3,min_periods=1).mean()
    return sma_data

# 累积平均平滑
def CumulativeMovingAverage(data):
    if isinstance(data,list):
        data = list_to_series(data)
        data = data.dropna()
    cma_data = data.expanding().mean()
    return cma_data

# 指数平均平滑
def ExponentialMovingAverage(data,alpha=0.1):
    if isinstance(data,list):
        data = list_to_series(data)
        data = data.dropna()
    ema_data = data.ewm(alpha=alpha,adjust=False).mean()
    return ema_data

# Amplitude Spectrum
def calcuate_amplitude_spectrum(data,mode='fft',normalization=False):
    if  not isinstance(data,list):
        data =series_to_list(data) 
    data_array = np.array(data)
    N = len(data_array)
    # fft
    if mode=='fft':
        Y = fft(data_array)  # format: a+bj , and the amplitutde means the norm length , angle is the complex angle
        positive_part_from_fft = Y[:Y.size//2]
        abs_positive_y = np.abs(positive_part_from_fft)
        abs_y= np.abs(Y)
        angle_y = np.angle(Y)
    #shift fft
    elif mode=='ffshift':
        Y =fftshift(data_array)
        positive_part_from_fft = Y[:Y.size//2]
        abs_positive_y = np.abs(positive_part_from_fft)
        abs_y= np.abs(Y)
        angle_y = np.angle(Y)
    else:
        raise NotImplementedError   
    if normalization:
        abs_y = abs_y/N
        Y = Y/N
    
    return[Y,abs_y,angle_y]

# Power Spectrum Calculation
def calculate_power_spectrum(data,num_fft=1024,use_correlate=False,draw=False):
    if  not isinstance(data,list):
        data =series_to_list(data) 
    data_array = np.array(data)
    N = len(data_array)
    num_fft = num_fft
    # Power Spectrum
    if not use_correlate:
        Y = fft(data,num_fft)
        Y = np.abs(Y)
        ps = Y**2/num_fft
    else:
        cor_x = np.correlate(data,data,'same')
        cor_x = fft(cor_x,num_fft)
        ps_cor = np.abs(cor_x)
        ps_cor = ps_cor/np.max(ps_cor)
        ps = ps_cor
    if draw:
        plt.plot(20*np.log10(ps[:num_fft//2]))
        plt.show()

    return ps

    
# Cepstrum Calculation
def calculate_cepstrum(data,num_fft):
    if  not isinstance(data,list):
        data =series_to_list(data) 
    data_array = np.array(data)
    N = len(data_array)
    num_fft = num_fft
    specturm = np.fft.fft(data,num_fft)
    ceps = np.fft.ifft(np.log(np.abs(specturm))).real
    return ceps

# Cross-Corrleation Matrix
def cross_corrletion_matrix(data1,data2):
    data1 = np.array(data1)
    data2 = np.array(data2)
    return np.cov(data1,data2)

# Compute the similarity
def compute_the_R(data1,data2):
    cor_martix = cross_corrletion_matrix(data1=data1,data2=data2)
    r = cor_martix[0][0]*cor_martix[1][1]/(cor_martix[0][1]*cor_martix[1][0])
    return r

def spectrum(x, T):
    amp_spec = np.abs(np.fft.fft(x))
    power_spec = amp_spec ** 2

    freqs = np.fft.fftfreq(len(x), T)
    idx = np.argsort(freqs)

    return amp_spec[idx], power_spec[idx], freqs[idx]

def draw_spectrum(data_list,city_name,folder_name):
    T = 3600

    amp_spec, power_spec, freq = spectrum(data_list, T)

    print('Max amp in spectrum:', np.max(amp_spec))
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    x = list(range(len(data_list)))
    y = data_list
    plt.xlabel('Hours')
    plt.ylabel('Observation')
    plt.plot(x, y)
    plt.title("{} Tide Data in March".format(city_name))

    plt.subplot(132)
    x = freq[360:]
    y = amp_spec[360:]
    # set 0 to 0Hz (DC component)
    y[0] = 0
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Intensity')
    plt.plot(x, y)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    plt.title("{} Amp Specturm".format(city_name))

    plt.subplot(133)
    y = power_spec[360:]
    # set 0 to 0Hz (DC component)
    y[0] = 0
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Intensity')
    plt.plot(x, y)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    plt.title("{} Power Specturm".format(city_name))
    plt.savefig("{}/{}_specturm_graphs.png".format(folder_name, city_name))
    plt.show()



def remove_bias_and_trend(x):
    mean = np.mean(x)
    xp = x - mean
    xp = detrend(xp)
    return xp

def auto_corr(x, lags):
    xp = remove_bias_and_trend(x)
    var = np.var(xp)

    corr = np.correlate(xp, xp, 'full')[len(x) - 1:]

    # divide the correlation by b-a+1 for each lag
    corr = np.asarray([corr[i] / (len(x) - i) for i in range(len(x))])
    corr = corr / var

    return corr[:lags], xp


def cross_corr(x, y, lags):
    xp = remove_bias_and_trend(x)
    yp = remove_bias_and_trend(y)
    var_x = np.var(xp)
    var_y = np.var(yp)

    corr = np.correlate(xp, yp, 'full')[len(x) - 1:]
    corr = np.asarray([corr[i] / (len(x) - i) for i in range(len(x))])
    corr = corr / np.sqrt(var_x * var_y)
    return corr[:lags], xp, yp

def draw_acfs(data_list, lags=300):

    acf, detrended_x = auto_corr(data_list, lags)
    print('The shape of acf: ', acf.shape)

    plt.subplot(131)
    x = list(range(len(data_list)))
    y = data_list
    plt.xlabel('Hours')
    plt.ylabel('Observation')
    plt.plot(x, y)

    plt.subplot(132)
    x = list(range(len(detrended_x)))
    y = detrended_x
    plt.xlabel('Hours')
    plt.ylabel('Observation')
    plt.plot(x, y)

    plt.subplot(133)
    x = list(range(lags))
    y = acf
    print('Min and max of acf: ', np.min(acf), np.max(acf))
    plt.plot(x, y)
    ax = plt.gca()
    ax.set_xlim(0, lags)
    maloc = plt.MultipleLocator(6)
    ax.xaxis.set_major_locator(maloc)
    plt.grid()
    plt.xlabel('Time lag')
    plt.ylabel('ACF')

    plt.show()


def draw_cross_corrs(data_list_x, data_list_y, lags=300):

    ccf, xp, yp = cross_corr(data_list_x, data_list_y, lags)
    print('The shape of ccf: ', ccf.shape)

    plt.subplot(131)
    x = list(range(len(xp)))
    y = xp
    plt.xlabel('Hours')
    plt.ylabel('Observation')
    plt.plot(x, y)

    plt.subplot(132)
    x = list(range(len(yp)))
    y = yp
    plt.xlabel('Hours')
    plt.ylabel('Observation')
    plt.plot(x, y)

    plt.subplot(133)
    x = list(range(lags))
    y = ccf
    print('Min and max of acf: ', np.min(ccf), np.max(ccf))
    plt.plot(x, y)
    ax = plt.gca()
    ax.set_xlim(0, lags)
    maloc = plt.MultipleLocator(6)
    ax.xaxis.set_major_locator(maloc)
    plt.grid()
    plt.xlabel('Time lag')
    plt.ylabel('Cross Correlation')

    plt.show()


def cross_corr(x, y, lags):
    print(lags)
    xp = remove_bias_and_trend(x)
    yp = remove_bias_and_trend(y)
    var_x = np.var(xp)
    var_y = np.var(yp)

    corr = np.correlate(xp, yp, 'full')[len(x) - 1:]
    corr = np.asarray([corr[i] / (len(x) - i) for i in range(len(x))])
    corr = corr / np.sqrt(var_x * var_y)

    return corr[:lags], xp, yp

def draw_cross_corr(data_list_x, data_list_y, lags=300, saved_cities=None,saved_folder=None):

    ccf, xp, yp = cross_corr(data_list_x, data_list_y, 300)
    print('The shape of ccf: ', ccf.shape)
    plt.figure(figsize=(15,4))

    plt.subplot(131)
    x = list(range(len(xp)))
    y = xp
    plt.xlabel('Hours')
    plt.ylabel('Observation')
    plt.title("{} Tide Data".format(saved_cities[0]))
    plt.plot(x, y)

    plt.subplot(132)
    x = list(range(len(yp)))
    y = yp
    plt.xlabel('Hours')
    plt.ylabel('Observation')
    plt.title("{} Tide Data".format(saved_cities[1]))
    plt.plot(x, y)

    plt.subplot(133)
    x = list(range(96))
    y = ccf[:96]
    print('Min and max of acf: ', np.min(ccf), np.max(ccf))
    plt.title("{}-{} cross correlation graph".format(saved_cities[0], saved_cities[1]))
    plt.plot(x, y)
    ax = plt.gca()
    ax.set_xlim(0, 96)
    maloc = plt.MultipleLocator(6)
    ax.xaxis.set_major_locator(maloc)
    plt.grid()
    plt.xlabel('Time lag')
    plt.ylabel('Cross Correlation')

    if saved_cities is not None:
        plt.savefig("{}/{}_{}cross.png".format(saved_folder,saved_cities[0],saved_cities[1]))
    plt.show()




