import numpy as np
import pandas as pd
import datetime as dt

from scipy.signal import welch
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from scipy.fftpack import fft

from siml.detect_peaks import detect_peaks

def get_values(y_values, T, N, f_s):
    y_values = y_values
    x_values = [(1/f_s) * kk for kk in range(0,len(y_values))]
    return x_values, y_values

def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def get_psd_values(y_values, T, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]
 
def get_autocorr_values(y_values, T, N, f_s):
    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])
    return x_values, autocorr_values
    
def get_first_n_peaks(x,y,no_peaks=5):
    x_, y_ = list(x), list(y)
    if len(x_) >= no_peaks:
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks-len(x_)
        return x_ + [0]*missing_no_peaks, y_ + [0]*missing_no_peaks
    
def get_features(x_values, y_values, mph):
    indices_peaks = detect_peaks(y_values, mph=mph)
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks], y_values[indices_peaks])
    return peaks_x + peaks_y
 
def extract_features_labels(dataset, labels, T, N, f_s, denominator):
    percentile = 5
    list_of_features = []
    list_of_labels = []
    for signal_no in range(0, len(dataset)):
        features = []
        list_of_labels.append(labels[signal_no])
        for signal_comp in range(0,dataset.shape[2]):
            signal = dataset[signal_no, :, signal_comp]
            
            signal_min = np.nanpercentile(signal, percentile)
            signal_max = np.nanpercentile(signal, 100-percentile)
            #ijk = (100 - 2*percentile)/10
            mph = signal_min + (signal_max - signal_min)/denominator
            
            features += get_features(*get_psd_values(signal, T, N, f_s), mph)
            features += get_features(*get_fft_values(signal, T, N, f_s), mph)
            features += get_features(*get_autocorr_values(signal, T, N, f_s), mph)
        list_of_features.append(features)
    return np.array(list_of_features), np.array(list_of_labels)

def plot_yvalues(ax, xvalues, yvalues, plot_original=True, polydeg=2):
    z2 = np.polyfit(xvalues, yvalues, polydeg)
    p2 = np.poly1d(z2)
    yvalues_trend = p2(xvalues)
    yvalues_detrended = yvalues - yvalues_trend
    ax.plot(xvalues, yvalues_detrended,  color='green', label='detrended')
    ax.plot(xvalues, yvalues_trend, linestyle='--', color='k', label='trend')
    if plot_original:
        ax.plot(xvalues, yvalues, color='skyblue',label='original')
    ax.legend(loc='upper left', bbox_to_anchor=(-0.09, 1.078),framealpha=1)
    ax.set_yticks([])
    return yvalues_detrended

def plot_fft_psd(ax, yvalues_detrended, plot_psd=True, annotate_peaks=True, max_peak=0.5):
    assert max_peak > 0 and max_peak <= 1, "max_peak should be between 0 and 1"
    fft_x_ = np.fft.fftfreq(len(yvalues_detrended))
    fft_y_  = np.fft.fft(yvalues_detrended);
    fft_x = fft_x_[:len(fft_x_)//2]
    fft_y = np.abs(fft_y_[:len(fft_y_)//2])
    psd_x, psd_y = welch(yvalues_detrended)
    
    mph = np.nanmax(fft_y)*max_peak
    indices_peaks = detect_peaks(fft_y, mph=mph)
    peak_fft_x, peak_fft_y = fft_x[indices_peaks], fft_y[indices_peaks]
    
    ax.plot(fft_x, fft_y, label='FFT')
    if plot_psd:
        axb = ax.twinx()
        axb.plot(psd_x, psd_y, color='red',linestyle='--', label='PSD')
    if annotate_peaks:
        for ii in range(len(indices_peaks)):
            x, y = peak_fft_x[ii], peak_fft_y[ii]
            T = 1/x
            text = "  f = {:.2f}\n  T = {:.2f}".format(x,T)
            ax.annotate(text, (x,y),va='top')
    
    lines, labels = ax.get_legend_handles_labels()
    linesb, labelsb = axb.get_legend_handles_labels()
    ax.legend(lines + linesb, labels + labelsb, loc='upper left')
    
    ax.set_yticks([])
    axb.set_yticks([])
    return fft_x_, fft_y_

def plot_predicted(ax, yvalues_train, yvalues_test, no_harmonics='all', deg_polyfit=2):
    predicted = reconstruct_from_fft(yvalues_train, no_harmonics=no_harmonics, deg_polyfit=deg_polyfit, additional_samples=len(yvalues_test))
    predicted = predicted_all[len(yvalues_train):]
    no_harmonics_75 = int(0.25*len(yvalues_train))
    no_harmonics_50 = int(0.15*len(yvalues_train))
    predicted_75 = reconstruct_from_fft(yvalues_train, no_harmonics=no_harmonics_75, deg_polyfit=deg_polyfit, additional_samples=len(yvalues_test))
    predicted_75 = predicted_75[len(yvalues_train):]
    predicted_50 = reconstruct_from_fft(yvalues_train, no_harmonics=no_harmonics_50, deg_polyfit=deg_polyfit, additional_samples=len(yvalues_test))
    predicted_50 = predicted_50[len(yvalues_train):]
    ax.plot(yvalues_test, label='true yvalues', linewidth=2)
    ax.plot(predicted_all, label='predicted (all harmonics)')
    ax.plot(predicted_75, label='predicted ({} harmonics)'.format(no_harmonics_75))
    ax.plot(predicted_50, label='predicted ({} harmonics)'.format(no_harmonics_50))
    ax.set_yticks([])
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.078))
    
def fill_missing_dates(df__, datecol='date'):
    df__.index = df__[datecol].values
    
    start = str(df__[datecol].min())
    end = str(df__[datecol].max())
    missing_indices = pd.date_range(start=start, end=end, freq='D').difference(df__[datecol])
    df_extra = pd.DataFrame(columns=df__.columns, index=missing_indices)
    df_extra.loc[:,datecol] = df_extra.index.values
    df_extra.loc[:,datecol] = df_extra[datecol].apply(lambda x: x.date())
    df_extra.index = df_extra[datecol].values

    df__ = df__.append(df_extra).sort_values([datecol]).reset_index()
    df__ = df__.fillna(0)
    return df__

def construct_fft(yvalues, deg_polyfit, real_abs_only=True):
    N = len(yvalues)
    xvalues = np.arange(N)
    
    # we calculate the trendline and detrended signal with polyfit
    z2 = np.polyfit(xvalues, yvalues, deg_polyfit)
    p2 = np.poly1d(z2)
    yvalues_trend = p2(xvalues)
    yvalues_detrended = yvalues - yvalues_trend
    
    # The fourier transform and the corresponding frequencies
    fft_y = np.fft.fft(yvalues_detrended)
    fft_x = np.fft.fftfreq(N)
    if real_abs_only:
        fft_x = fft_x[:len(fft_x)//2]
        fft_y = np.abs(fft_y[:len(fft_y)//2])
    return fft_x, fft_y, p2

def get_integer_no_of_periods(yvalues, fft_x, fft_y, frac=1.0, mph=0.4):
    N = len(yvalues)
    fft_y_real = np.abs(fft_y[:len(fft_y)//2])
    fft_x_real = fft_x[:len(fft_x)//2]
    
    mph = np.nanmax(fft_y_real)*mph
    indices_peaks = detect_peaks(fft_y_real, mph=mph)
    peak_fft_x = fft_x_real[indices_peaks]
    main_peak_x = peak_fft_x[0]
    T = int(1/main_peak_x)
    
    no_integer_periods_all = N//T
    no_integer_periods_frac = int(frac*no_integer_periods_all)
    no_samples = T*no_integer_periods_frac
    
    yvalues_ = yvalues[-no_samples:]
    xvalues_ = np.arange(len(yvalues))
    return xvalues_, yvalues_

def restore_signal_from_fft(fft_x, fft_y, N, extrapolate_with, frac_harmonics):
    xvalues_full = np.arange(0, N + extrapolate_with)
    restored_sig = np.zeros(N + extrapolate_with)
    indices = list(range(N))
    
    # The number of harmonics we want to include in the reconstruction
    indices.sort(key = lambda i: np.absolute(fft_x[i]))
    max_no_harmonics = len(fft_y)
    no_harmonics = int(frac_harmonics*max_no_harmonics)
        
    for i in indices[:1 + no_harmonics * 2]:
        ampli = np.absolute(fft_y[i]) / N
        phase = np.angle(fft_y[i])
        restored_sig += ampli * np.cos(2 * np.pi * fft_x[i] * xvalues_full + phase)
    # return the restored signal plus the previously calculated trend
    return restored_sig

def reconstruct_from_fft(yvalues, 
                         frac_harmonics=1.0, 
                         deg_polyfit=2, 
                         extrapolate_with=0, 
                         fraction_signal = 1.0,
                         mph = 0.4):
    N_original = len(yvalues)
    
    fft_x, fft_y, p2 = construct_fft(yvalues, deg_polyfit, real_abs_only=False)
    xvalues, yvalues = get_integer_no_of_periods(yvalues, fft_x, fft_y, frac=fraction_signal, mph=mph)
    fft_x, fft_y, p2 = construct_fft(yvalues, deg_polyfit, real_abs_only=False)
    N = len(yvalues)

    xvalues_full = np.arange(0, N + extrapolate_with)
    restored_sig = restore_signal_from_fft(fft_x, fft_y, N, extrapolate_with, frac_harmonics)
    restored_sig = restored_sig + p2(xvalues_full)
    return restored_sig[-extrapolate_with:]
