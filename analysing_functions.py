import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import scipy.io as scio
import pandas as pd
from scipy.signal import hilbert

"""
def envelope(signal,fs):
     analytic_signal = hilbert(signal)
     amplitude_envelope = np.abs(analytic_signal)
     instantaneous_phase = np.unwrap(np.angle(analytic_signal))
     instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * fs

     return amplitude_envelope, instantaneous_frequency
"""
"""
def butter_bandpass_filter_allchs(data_buffer,fs, low_f, high_f):
    filtered_data = []
    nyq = 0.5 * fs
    lowcut = low_f
    highcut = high_f
    low = lowcut / nyq
    high = highcut / nyq
    order = 3
    b, a = signal.butter(order, [low, high], btype='band')
    print(data_buffer.shape)

    if (data_buffer.shape[0]>1):
        for i in range(8):
            channel = data_buffer[i]
            filtered = signal.filtfilt(b, a, channel)
            filtered_data.append(filtered)
    else:
        filtered_data = signal.filtfilt(b, a, data_buffer)

    filtered_data = np.array(filtered_data)
    return filtered_data
"""
def butter_bandpass_filter(data_buffer,fs, low_f, high_f):
    nyq = 0.5 * fs
    lowcut = low_f
    highcut = high_f
    low = lowcut / nyq
    high = highcut / nyq
    order = 3

    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data_buffer)

    return filtered_data

"""
def butter_lowpass_filter(data_buffer,fs, low_f):
    nyq = 0.5 * fs
    normal_cutoff = low_f / nyq
    order = 3

    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data_buffer)

    return filtered_data
"""
"""
def butter_notch_filter(data_buffer,fs):
    nyq = 0.5 * fs
    lowcut = 55.0
    highcut = 65.0
    low = lowcut / nyq
    high = highcut / nyq
    order = 3

    b, a = signal.butter(order, [low, high], btype='band')

    filtered_data = signal.filtfilt(b, a, data_buffer)

    return filtered_data
"""
def bandpower(data, sf, band, window_sec=None, relative=False):

    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res) #Absolute or relative band power.

    return bp

def band_powers_channelwise(eEEG_nparr, fs, num_channels, eEEG_mean, win, sub_left_title, sub_right_title, subs_dta,subs_dtr,subs_bta,subs_btr,subs_dba,subs_dbr,subs_atr,subs_ata,subs_ttr,subs_tta,subs_gtr,subs_gta, win_sec = 4):
    alpha_eEEG = []
    beta_eEEG = []
    theta_eEEG =[]
    gamma_eEEG = []

    periodgram_raw = []
    periodgram_beta = []
    welch_raw = []
    welch_beta = []

    dta = np.zeros(num_channels, dtype = float)
    dtr = np.zeros(num_channels, dtype = float)
    bta = np.zeros(num_channels, dtype = float)
    btr = np.zeros(num_channels, dtype = float)
    dba = np.zeros(num_channels, dtype = float)
    dbr = np.zeros(num_channels, dtype = float)
    ata = np.zeros(num_channels, dtype = float)
    atr = np.zeros(num_channels, dtype = float)
    tta = np.zeros(num_channels, dtype = float)
    ttr = np.zeros(num_channels, dtype = float)
    gta = np.zeros(num_channels, dtype = float)
    gtr = np.zeros(num_channels, dtype = float)

    plt.Figure()
    for i in range(num_channels):
        eEEG_nparr[:,i] = eEEG_nparr[:,i]-eEEG_mean
        #detrended_data = signal.detrend(data)

        eEEG_nparr[:,i] = butter_bandpass_filter(eEEG_nparr[:,i],fs, 0.5, 60.0)
        freqs, prdgmr = signal.periodogram(eEEG_nparr[:,i], fs, window = 'hamming' )
        freqs_welch, welchr = signal.welch(eEEG_nparr[:,i], fs, nperseg=win)
            # mtr, freqs_mt = psd_array_multitaper(eEEG_nparr[:,i], fs, adaptive=True,normalization='full', verbose=0)

                # Delta/Total ratio based on the absolute power
        dt = bandpower(eEEG_nparr[:,i], fs, [0.5, 4], win_sec)

                # Delta/Total ratio based on the relative power
        dt_rel = bandpower(eEEG_nparr[:,i], fs, [0.5, 4], win_sec, True)

                # beta/total ratio based on the absolute power
        bt = bandpower(eEEG_nparr[:,i], fs, [12, 30], win_sec)

                # beta/total ratio based on the relative power
        bt_rel = bandpower(eEEG_nparr[:,i], fs, [12, 30], win_sec, True)


                # Delta/beta ratio based on the absolute power
        db = bandpower(eEEG_nparr[:,i], fs, [0.5, 4], win_sec) / bandpower(eEEG_nparr[:,i], fs, [12, 30], win_sec)

                # Delta/beta ratio based on the relative power
        db_rel = bandpower(eEEG_nparr[:,i], fs, [0.5, 4], win_sec, True) / bandpower(eEEG_nparr[:,i], fs, [12, 30], win_sec, True)

                # Delta/beta ratio based on the absolute power
        at = bandpower(eEEG_nparr[:,i], fs, [8.0, 12.0], win_sec)

                # Delta/beta ratio based on the relative power
        at_rel = bandpower(eEEG_nparr[:,i], fs, [8.0, 12.0], win_sec, True)


                # Delta/beta ratio based on the absolute power
        tt = bandpower(eEEG_nparr[:,i], fs, [4.0, 8.0], win_sec)

                # Delta/beta ratio based on the relative power
        tt_rel = bandpower(eEEG_nparr[:,i], fs, [4.0, 8.0], win_sec, True)

                # Delta/beta ratio based on the absolute power
        gt = bandpower(eEEG_nparr[:,i], fs, [30.0, 60.0], win_sec)

                # Delta/beta ratio based on the relative power
        gt_rel = bandpower(eEEG_nparr[:,i], fs, [30.0, 60.0], win_sec, True)

        periodgram_raw.append(prdgmr)
        welch_raw.append(welchr)

        dta[i] = dt
        dtr[i] = dt_rel

        bta[i] = bt
        btr[i] = bt_rel

        dba[i] = db
        dbr[i] = db_rel

        ata[i] = at
        atr[i] = at_rel

        tta[i] = tt
        ttr[i] = tt_rel

        gta[i] = gt
        gtr[i] = gt_rel

        theta_eEEG.append(butter_bandpass_filter((eEEG_nparr[:,i]),fs, 4.0, 8.0))
        alpha_eEEG.append(butter_bandpass_filter((eEEG_nparr[:,i]),fs, 8.0, 12.0))
        beta_eEEG.append(butter_bandpass_filter((eEEG_nparr[:,i]),fs, 15.0, 30.0))
        freqs, prdgmb = signal.periodogram(np.array(beta_eEEG[i]), fs, window = 'hamming')
        freqs_welch, welchb = signal.welch(beta_eEEG[i], fs, nperseg=win)
        periodgram_beta.append(prdgmb)
        welch_beta.append(welchb)
        gamma_eEEG.append(butter_bandpass_filter((eEEG_nparr[:,i]),fs, 30.0, 62.0))

    subs_dta.append(dta)
    subs_dtr.append(dtr)
    subs_bta.append(bta)
    subs_btr.append(btr)
    subs_dba.append(dba)
    subs_dbr.append(dbr)
    subs_atr.append(atr)
    subs_ata.append(ata)
    subs_ttr.append(ttr)
    subs_tta.append(tta)
    subs_gtr.append(gtr)
    subs_gta.append(gta)

    alpha_eEEG = np.array(np.transpose(alpha_eEEG))
    beta_eEEG = np.array(np.transpose(beta_eEEG))
    theta_eEEG = np.array(np.transpose(theta_eEEG))
    gamma_eEEG = np.array(np.transpose(gamma_eEEG))
    periodgram_raw = np.array(np.transpose(periodgram_raw))
    periodgram_beta = np.array(np.transpose(periodgram_beta))
    welch_raw = np.array(np.transpose(welch_raw))
    welch_beta = np.array(np.transpose(welch_beta))

    fig, axes = plt.subplots(nrows=4,ncols=2, sharex=True)
    i=0
    for ax in axes:
         ax[0].plot(eEEG_nparr[:,i],label='raw-EEG')
         ax[0].plot(theta_eEEG[:,i],label='Theta-EEG')
         ax[0].plot(alpha_eEEG[:,i],label='Alpha-EEG')
         ax[0].plot(beta_eEEG[:,i],label='Beta-EEG')
         ax[0].plot(gamma_eEEG[:,i],label='Gamma-EEG')
         ax[0].set_title(sub_left_title)
         ax[0].legend()

         ax[1].plot(eEEG_nparr[:,i+1],label='raw-EEG')
         ax[1].plot(theta_eEEG[:,i+1],label='Theta-EEG')
         ax[1].plot(alpha_eEEG[:,i+1],label='Alpha-EEG')
         ax[1].plot(beta_eEEG[:,i+1],label='Beta-EEG')
         ax[1].plot(gamma_eEEG[:,i+1],label='Gamma-EEG')
         ax[1].set_title(sub_right_title)
         ax[1].legend()
         i+=1

    fig, axes = plt.subplots(nrows=4,ncols=2,sharex=True)
    i=0
    for ax in axes:
         ax[0].plot(freqs,periodgram_raw[:,i]/max(periodgram_raw[:,i]),label='raw-periodogram')
         ax[0].plot(freqs,periodgram_beta[:,i]/max(periodgram_beta[:,i]),label='beta-periodogram')
         ax[0].set_title(sub_left_title)
         ax[0].set_xlim(0, 50)
         ax[0].legend()

         ax[1].plot(freqs,periodgram_raw[:,i+1]/max(periodgram_raw[:,i+1]),label='raw-periodogram')
         ax[1].plot(freqs,periodgram_beta[:,i+1]/max(periodgram_beta[:,i+1]),label='beta-periodogram')
         ax[1].set_title(sub_right_title)
         ax[1].set_xlim(0, 50)
         ax[1].legend()
         i+=1

    fig, axes = plt.subplots(nrows=4,ncols=2,sharex=True)
    i=0
    for ax in axes:
         ax[0].plot(freqs_welch,welch_raw[:,i]/max(welch_raw[:,i]),label='raw-welch')
         ax[0].plot(freqs_welch,welch_beta[:,i]/max(periodgram_beta[:,i]),label='beta-welch')
         ax[0].set_title(sub_left_title)
         ax[0].set_xlim(0, 50)
         ax[0].legend()

         ax[1].plot(freqs_welch,welch_raw[:,i+1]/max(welch_raw[:,i+1]),label='raw-welch')
         ax[1].plot(freqs_welch,welch_beta[:,i+1]/max(welch_beta[:,i+1]),label='beta-welch')
         ax[1].set_title(sub_right_title)
         ax[1].set_xlim(0, 50)
         ax[1].legend()
         i+=1
         plt.pause(0.1)
         plt.close()

#    return subs_dta,subs_dtr,subs_bta,subs_btr,subs_dba,subs_dbr,subs_atr,subs_ata,subs_ttr,subs_tta,subs_gtr,subs_gta
    return subs_dtr,subs_btr,subs_dbr,subs_atr,subs_ttr,subs_gtr


def per_sub_band_power(subs_dta,subs_dtr,subs_bta,subs_btr,subs_dba,subs_dbr,subs_atr,subs_ata,subs_ttr,subs_tta,subs_gtr,subs_gta, path_eachsub, cols_names, num_channels, index, win, fs = 1000, filename = "syncEEG.mat", mat_var = 'new_EEG'):
        sub_left_title = index + "-Left"
        sub_right_title = index + "-Right"

        mat_filepath = os.path.join(path_eachsub,index,filename)
        print(mat_filepath)
        mat = scio.loadmat(mat_filepath)

        EEG_channels = pd.DataFrame(mat[mat_var])
        eEEG_channels = EEG_channels[cols_names]

        ## Converting the EEG channels into numpy array
        eEEG_nparr = np.array(eEEG_channels)
        eEEG_mean = eEEG_nparr.mean(axis=1)
        eEEG_std = np.std(eEEG_nparr,axis=0)

        eEEG_nparr = (eEEG_nparr - eEEG_mean)/eEEG_std

#        [subs_dta,subs_dtr,subs_bta,subs_btr,subs_dba,subs_dbr,subs_atr,subs_ata,subs_ttr,subs_tta,subs_gtr,subs_gta] = band_powers_channelwise(eEEG_nparr, fs, num_channels, eEEG_mean, win, sub_left_title, sub_right_title, subs_dta,subs_dtr,subs_bta,subs_btr,subs_dba,subs_dbr,subs_atr,subs_ata,subs_ttr,subs_tta,subs_gtr,subs_gta, win_sec = 4)
        [subs_dtr,subs_btr,subs_dbr,subs_atr,subs_ttr,subs_gtr] = band_powers_channelwise(eEEG_nparr, fs, num_channels, eEEG_mean, win, sub_left_title, sub_right_title, subs_dta,subs_dtr,subs_bta,subs_btr,subs_dba,subs_dbr,subs_atr,subs_ata,subs_ttr,subs_tta,subs_gtr,subs_gta, win_sec = 4)

        subs_dta = np.array(subs_dta)
        subs_dtr = np.array(subs_dtr)
        subs_bta = np.array(subs_bta)
        subs_btr = np.array(subs_btr)
        subs_dba = np.array(subs_dba)
        subs_dbr = np.array(subs_dbr)
        subs_atr = np.array(subs_atr)
        subs_ata = np.array(subs_ata)
        subs_ttr = np.array(subs_ttr)
        subs_tta = np.array(subs_tta)
        subs_gtr = np.array(subs_gtr)
        subs_gta = np.array(subs_gta)

#        return subs_dta,subs_dtr,subs_bta,subs_btr,subs_dba,subs_dbr,subs_atr,subs_ata,subs_ttr,subs_tta,subs_gtr,subs_gta
        return subs_dtr,subs_btr,subs_dbr,subs_atr,subs_ttr,subs_gtr