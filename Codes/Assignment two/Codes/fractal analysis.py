import numpy as np
from scipy.signal import detrend
from pylab import plt
import matplotlib.ticker as mtick
from utils import read_csv_data
import hfd

def create_image(csv_path):
    csv_data,_,_ = read_csv_data(csv_path)
    plt.figure()
    plt.plot(csv_data)
    plt.axis('off')

    plt.savefig('wind_data.png', bbox_inches='tight', dpi=500)


def rgb2gray(rgb):
    gray = rgb[..., 2]
    gray = gray < 1
    plt.imshow(np.asarray(gray, dtype=np.float32))
    plt.show()
    return gray


def box_count(image_path='wind_data.png'):
    image = rgb2gray(plt.imread(image_path))

    # finding all the non-zero pixels
    pixels = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > 0:
                pixels.append((i, j))

    Lx = image.shape[1]
    Ly = image.shape[0]
    print(Lx, Ly)
    pixels = np.array(pixels)
    print(pixels.shape)

    # computing the fractal dimension
    # considering only scales in a logarithmic list
    scales = np.logspace(0.01, 1, num=20, endpoint=False, base=2)
    Ns = []
    # looping over several scales
    for scale in scales:
        print("======= Scale :", scale)
        # computing the histogram
        H, edges = np.histogramdd(pixels, bins=(np.arange(0, Lx, scale), np.arange(0, Ly, scale)))
        Ns.append(np.sum(H > 0))

    # linear fit, polynomial of degree 1
    coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)

    plt.figure()
    plt.title("Log-Log Linear Fit of Kyoto")
    plt.plot(np.log(scales), np.log(Ns), 'o', mfc='none')
    plt.plot(np.log(scales), np.polyval(coeffs, np.log(scales)))
    plt.xlabel('log $\epsilon$')
    plt.ylabel('log N')
    plt.savefig("../pics/log_log_fit.png")
    plt.show()

    print("The Hausdorff dimension is", -coeffs[0])  # the fractal dimension is the OPPOSITE of the fitting coefficient
    np.savetxt("scaling.txt", list(zip(scales, Ns)))



def spectrum(x, T):
    amp_spec = np.abs(np.fft.fft(x))
    power_spec = amp_spec ** 2

    freqs = np.fft.fftfreq(len(x), T)
    idx = np.argsort(freqs)

    return amp_spec[idx], power_spec[idx], freqs[idx]


def draw_spectrum(data_list):
    T = 3600

    amp_spec, power_spec, freq = spectrum(data_list, T)

    print('Max amp in spectrum: {np.max(amp_spec)}')
    plt.figure(figsize=(18,5))

    plt.subplot(131)
    x = list(range(len(data_list)))
    y = data_list
    plt.title("Observation wind data of Kyoto")
    plt.xlabel('Hours')
    plt.ylabel('Observation wind data of Kyoto')
    plt.plot(x, y,color='green')

    data_len = len(x)

    plt.subplot(132)
    plt.title("Power Spectrum of Wind ")
    x = freq[int(data_len / 2):]
    y = power_spec[int(data_len / 2):]

    # set 0 to 0Hz (DC component)
    y[0] = 0

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Intensity')
    plt.plot(x, y,color='orange')
    ax = plt.gca()

    x = x[1:]
    y = y[1:]

    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    coeffs = np.polyfit(np.log(x), np.log(y), 1)
    beta = -coeffs[0]
    dimension = 1 + (3-beta) / 2
    print(beta)
    print("The fractal dimension is", dimension)


    plt.subplot(133)
    plt.title("the Curve of log(power-spectrum) and log(frequency)")
    plt.scatter(np.log(x), np.log(y),marker='o',s=10,c=list(range(len(x))))
    # plt.plot(np.log(x), np.log(y), 'o', mfc='none')
    plt.plot(np.log(x), np.polyval(coeffs, np.log(x)))
    plt.xlabel('log freq')
    plt.ylabel('log intensity')
    plt.savefig("../pics/kyoto_wind.png")

    plt.show()


if __name__ == '__main__':
    # data of kyoto
    filename='../Kyoto_wind.csv'
    wind_list, time_list, data_len = read_csv_data(filename=filename)
    print(hfd.hfd(wind_list))
    draw_spectrum(wind_list)
    create_image(filename)
    box_count()

