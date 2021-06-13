from  utils import read_csv_data
import  numpy as np
import  matplotlib.pyplot as plt
import  pywt
from matplotlib.ticker import FuncFormatter

def changey(temp, position):
    sampling_period = 3600
    freq = pywt.scale2frequency('morl', temp+1) / sampling_period
    freq = freq
    freq = '%.2e' % freq
    return freq

def wavelet_transform(data, scales):
    coef, freqs = pywt.cwt(data, scales, 'morl')
    print("Wavelet transform is Done")
    return coef, freqs

if __name__ =="__main__":
    filename = '../Naha_tem.csv'
    temp_list, time_list, data_len = read_csv_data(filename=filename)

    # For the Wavelet transform
    sampling_period = 3600
    scales = range(1500, 1600)
    coef, freqs = wavelet_transform(temp_list, scales)

    # Show the original_data
    plt.figure(figsize=(10, 4), dpi=80)
    plt.title("Original data of the Naha Temperature from 2020/06-2021/06")
    plt.xlabel('Time (per hour)', fontsize=8)
    plt.ylabel('Temperature (Celsius degree)', fontsize=12)
    plt.plot(temp_list,label= "Naha",color='green')
    plt.legend()
    plt.savefig("../pics/Naha_temp.png")
    plt.show()

    # Show the wavelet_transform matrix
    plt.figure(figsize=(16, 5), dpi=80)  # 图片长宽和清晰度
    plt.subplot(121)
    plt.title("WaveLet Transform Matrix of Naha")
    plt.xlabel('Time (per hour)', fontsize=10)
    plt.ylabel('Freq (Hz)', fontsize=10)

    plt.imshow(coef, cmap='rainbow', aspect='auto')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(changey))
    # plt.savefig("../pics/Tokyo_wavelet_transform.png")
    # plt.show()

    # Show the wavelet_transform matrix
    plt.subplot(122)
    plt.title("WaveLet Power Matrix of Naha")
    plt.xlabel('Time (per hour)', fontsize=10)
    plt.ylabel('Freq ( Hz)', fontsize=10)

    plt.imshow(coef**2, cmap='rainbow', aspect='auto')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(changey))
    plt.savefig("../pics/local/Naha_wavelet_transform0.png")
    plt.show()


