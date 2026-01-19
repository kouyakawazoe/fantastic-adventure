import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

class EMGAnalyzer:
    """
    筋電図（EMG）解析用クラス
    以下の8つの重要指標の算出をサポートします：
    MuscleSynergy, CCI, Onset, Duration, IEMG, Efficiency, Co-contraction, Switching, Asymmetry Index
    """
    def __init__(self, emg_data, sampling_rate):
        """
        Parameters:
        emg_data (np.ndarray): 筋電図データ (時間 x チャンネル)
        sampling_rate (int): サンプリング周波数 (Hz)
        """
        self.raw_data = emg_data
        self.fs = sampling_rate
        self.processed_data = None

    def preprocess(self, lowcut=20, highcut=450):
        """ノイズ除去と整流（バンドパスフィルタ + 絶対値化）"""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(4, [low, high], btype='band')
        
        # フィルタリングして絶対値をとる（整流）
        self.processed_data = np.abs(filtfilt(b, a, self.raw_data, axis=0))
        return self.processed_data

    def get_onset_duration(self, channel, threshold_sd=3):
        """Onset（開始）とDuration（持続時間）の算出"""
        data = self.processed_data[:, channel] if self.processed_data is not None else self.raw_data[:, channel]
        
        baseline = data[:self.fs] # 最初の1秒をベースラインと仮定
        threshold = np.mean(baseline) + threshold_sd * np.std(baseline)
        active_indices = np.where(data > threshold)[0]
        
        if len(active_indices) == 0:
            return None, None
            
        onset = active_indices[0] / self.fs
        duration = (active_indices[-1] - active_indices[0]) / self.fs
        return onset, duration

    def calculate_cci(self, ch_agonist, ch_antagonist):
        """CCI (Co-contraction Index) の算出"""
        emg1 = self.processed_data[:, ch_agonist] if self.processed_data is not None else self.raw_data[:, ch_agonist]
        emg2 = self.processed_data[:, ch_antagonist] if self.processed_data is not None else self.raw_data[:, ch_antagonist]
        
        low_act = np.minimum(emg1, emg2)
        high_act = np.maximum(emg1, emg2)
        
        # CCI = (low / high) * (low + high) の積分
        cci = np.trapz((low_act / (high_act + 1e-6)) * (low_act + high_act))
        return cci

    def extract_synergies(self, n_components=4):
        """NMFによる筋シナジー（Muscle Synergy）抽出"""
        data = self.processed_data if self.processed_data is not None else self.raw_data
        model = NMF(n_components=n_components, init='random', random_state=0, max_iter=1000)
        W = model.fit_transform(data) # シナジー構造（各筋の寄与度）
        H = model.components_          # 時間的活動パターン
        return W, H

    def calculate_iemg(self, channel):
        """IEMG (積分筋電図) の算出"""
        data = self.processed_data[:, channel] if self.processed_data is not None else self.raw_data[:, channel]
        return np.trapz(data)

    def asymmetry_index(self, val_r, val_l):
        """Asymmetry Index (左右非対称性指数) の算出"""
        return abs(val_r - val_l) / (val_r + val_l) * 100

    def plot_channel(self, channel):
        """特定のチャンネルの波形とOnsetをプロット"""
        data = self.processed_data[:, channel] if self.processed_data is not None else self.raw_data[:, channel]
        onset, _ = self.get_onset_duration(channel)
        
        plt.figure(figsize=(12, 4))
        plt.plot(np.arange(len(data))/self.fs, data, label='EMG Signal')
        if onset:
            plt.axvline(onset, color='r', linestyle='--', label=f'Onset: {onset:.3f}s')
        plt.title(f"Channel {channel} Analysis")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()
