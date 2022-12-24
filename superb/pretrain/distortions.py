import os
import glob
import yaml
import torch
import librosa
import numpy as np
import random
import augment
import torchaudio
import pyroomacoustics as pra
import soundfile as sf
from tqdm import tqdm
from dataclasses import dataclass
from scipy import signal
from cmath import log10


np.random.seed(7122)
random.seed(7122)


class RandomPitchShift:
    def __init__(self, shift_max=300):
        self.shift_max = shift_max

    def __call__(self):
        return np.random.randint(-self.shift_max, self.shift_max)

class RandomClipFactor:
    def __init__(self, factor_min=0.0, factor_max=1.0):
        self.factor_min = factor_min
        self.factor_max = factor_max
    def __call__(self):
        return np.random.triangular(self.factor_min, self.factor_max, self.factor_max)

@dataclass
class RandomReverb:
    reverberance_min: int = 50
    reverberance_max: int = 50
    damping_min: int = 50
    damping_max: int = 50
    room_scale_min: int = 0
    room_scale_max: int = 100

    def __call__(self):
        reverberance = np.random.randint(self.reverberance_min, self.reverberance_max + 1)
        damping = np.random.randint(self.damping_min, self.damping_max + 1)
        room_scale = np.random.randint(self.room_scale_min, self.room_scale_max + 1)

        return [reverberance, damping, room_scale]

class SpecAugmentBand:
    def __init__(self, sample_rate, scaler):
        self.sample_rate = sample_rate
        self.scaler = scaler

    @staticmethod
    def freq2mel(f):
        return 2595. * np.log10(1 + f / 700)

    @staticmethod
    def mel2freq(m):
        return ((10.**(m / 2595.) - 1) * 700)

    def __call__(self):
        F = 27.0 * self.scaler
        melfmax = self.freq2mel(self.sample_rate / 2)
        meldf = np.random.uniform(0, melfmax * F / 256.)
        melf0 = np.random.uniform(0, melfmax - meldf)
        low = self.mel2freq(melf0)
        high = self.mel2freq(melf0 + meldf)
        return f'{high}-{low}'


class DistortionFactory:
    def __init__(self, distortion_types, distortion_cfg, data_type=None):
        self.distortion_cfg = distortion_cfg
        if data_type is None:
            ValueError(f"data_type for DistortionFactory initialization must be specified")
        if 'm' in distortion_types:
            self.musan_files = glob.glob(os.path.join(distortion_cfg["musan"], '**/**/*.wav'))
        if 'mn' in distortion_types:
            self.musan_noise_files = glob.glob(os.path.join(distortion_cfg["musan_noise"], '**/*.wav'))
        if 'ms' in distortion_types:
            self.musan_speech_files = glob.glob(os.path.join(distortion_cfg["musan_speech"], '**/*.wav'))
        if 'mm' in distortion_types:
            self.musan_music_files = glob.glob(os.path.join(distortion_cfg["musan_music"], '**/*.wav'))
        if 'dns' in distortion_types:
            self.dns_noise_files = glob.glob(os.path.join(distortion_cfg["DNS_noise"], '*.wav'))
            self.dns_rir_files = glob.glob(os.path.join(distortion_cfg["DNS_rir"], '**/*.wav'), recursive=True)
        if 'wham' in distortion_types:
            print(f'Using {data_type} set of WHAM!')
            if data_type in ['train', 'training', 'tr']: 
                self.wham_noise_files = glob.glob(os.path.join(distortion_cfg["wham"]['tr'], '*.wav'))
            elif data_type in ['dev', 'valid', 'validation', 'cv']: 
                self.wham_noise_files = glob.glob(os.path.join(distortion_cfg["wham"]['cv'], '*.wav'))
            elif data_type in ['test', 'testing', 'tt']:
                self.wham_noise_files = glob.glob(os.path.join(distortion_cfg["wham"]['tt'], '*.wav'))
        if 'fsd' in distortion_types:
            self.fsd50k_noise_files = glob.glob(os.path.join(distortion_cfg["fsd50k"], '*.wav'))


    def load_wav(self, filename, desired_sample_rate):
        audio_signal, sample_rate = librosa.load(filename, sr=None)
        if audio_signal.ndim > 1:
            audio_signal = audio_signal[:, 0]
        audio_signal = librosa.resample(audio_signal, sample_rate, desired_sample_rate)
        return audio_signal

    def _snr_coeff(self, snr, signal, noise):
        pwr_signal = sum(signal ** 2)
        pwr_noise = sum(noise ** 2)
        return (pwr_signal / pwr_noise * 10 ** (-snr / 10)) ** 0.5

    def add_gau_noise(self, signal, snr):
        noise = np.random.randn(signal.shape[0])
        coeff = self._snr_coeff(snr, signal, noise)
        noise *= coeff
        signal += noise
        return signal
        #return signal, noise

    def add_real_noise(self, signal, noise, snr):
        signal_len = signal.shape[0]
        noise_len = noise.shape[0]
        if signal_len <= noise_len:
            start = random.randint(0, noise_len - signal_len)
            noise = noise[start: start+signal_len]
        else:
            n_repeat = signal_len // noise_len + 1
            noise = np.repeat(noise, n_repeat)
            noise = noise[: signal_len]
        coeff = self._snr_coeff(snr, signal, noise)
        noise *= coeff
        signal += noise
        return signal
        #return signal, noise

    def add_real_rir(self, wav, rir, snr):
        coeff = self._snr_coeff(snr, wav, rir)
        rir *= coeff
        reverb_speech = signal.fftconvolve(wav, rir, mode="full")
        reverb_speech = reverb_speech[0 : wav.shape[0]]
        return reverb_speech
    
    def add_distortion(self, signal, noise_type, sample_rate=None):
        snr = random.choice(range(10, 20, 1))
        if noise_type == 'g':
            return self.add_gau_noise(signal, snr)
        elif noise_type == 'm':
            noise_path = random.choice(self.musan_files)
            noise = self.load_wav(noise_path, sample_rate)
        elif noise_type == 'mn':
            noise_path = random.choice(self.musan_noise_files)
            noise = self.load_wav(noise_path, sample_rate)
            return self.add_real_noise(signal, noise, snr)
        elif noise_type == 'ms':
            noise_path = random.choice(self.musan_speech_files)
            noise = self.load_wav(noise_path, sample_rate)
            return self.add_real_noise(signal, noise, snr)
        elif noise_type == 'mm':
            noise_path = random.choice(self.musan_music_files)
            noise = self.load_wav(noise_path, sample_rate)
            return self.add_real_noise(signal, noise, snr)
        elif noise_type == 'wham':
            noise_path = random.choice(self.wham_noise_files)
            noise = self.load_wav(noise_path, sample_rate)
            return self.add_real_noise(signal, noise, snr)
        elif noise_type == 'fsd':
            noise_path = random.choice(self.fsd50k_noise_files)
            noise = self.load_wav(noise_path, sample_rate)
            return self.add_real_noise(signal, noise, snr)
        elif noise_type == 'dns':
            noise_path = random.choice(self.dns_noise_files)
            noise = self.load_wav(noise_path, sample_rate)
            rir_path = random.choice(self.dns_rir_files)
            rir = self.load_wav(rir_path, sample_rate)
            rir_snr = random.choice(range(10, 20, 1))
            rir_signal = self.add_real_rir(signal, rir, rir_snr)
            return self.add_real_noise(rir_signal, noise, snr)  
        elif noise_type == 'r' or noise_type == 'b' or noise_type == 'p' or noise_type == 't' or noise_type == 'clip':
            chain = augment.EffectChain()
            if noise_type == 'b':
                chain = chain.sinc('-a', '120', SpecAugmentBand(sample_rate, self.distortion_cfg["bandrej"]["band_scaler"]))
            elif noise_type == 'p':
                pitch_randomizer = RandomPitchShift(self.distortion_cfg["pitch"]["pitch_shift_max"])
                if self.distortion_cfg["pitch"]["pitch_quick"]:
                    chain = chain.pitch('-q', pitch_randomizer).rate('-q', sample_rate)
                else:
                    chain = chain.pitch(pitch_randomizer).rate(sample_rate)
            elif noise_type == 'r':
                randomized_params = RandomReverb(self.distortion_cfg["reverb"]["reverberance_min"], self.distortion_cfg["reverb"]["reverberance_max"], 
                                    self.distortion_cfg["reverb"]["damping_min"], self.distortion_cfg["reverb"]["damping_max"], self.distortion_cfg["reverb"]["room_scale_min"], self.distortion_cfg["reverb"]["room_scale_max"])
                chain = chain.reverb(randomized_params).channels()
            elif noise_type == 't':
                chain = chain.time_dropout(max_seconds=self.distortion_cfg["time_drop"]["t_ms"] / 1000.0)
            elif noise_type == 'clip':
                chain = chain.clip(RandomClipFactor(self.distortion_cfg["clip"]["clip_min"], self.distortion_cfg["clip"]["clip_max"])) 
            
            ori_signal_len = len(signal)    
            ori_signal = torch.Tensor(signal).unsqueeze(0)
            
            signal = chain.apply(ori_signal, 
                    src_info=dict(rate=sample_rate, length=ori_signal.size(1), channels=ori_signal.size(0)),
                    target_info=dict(rate=sample_rate, length=0)
            )
            signal_len = signal.size(1)
            # deal with inconsistent length of distorted and original speech
            if signal_len > ori_signal_len:
                signal = signal[:, :ori_signal_len].squeeze(0)
            elif signal_len < ori_signal_len:
                signal = torch.cat((signal, ori_signal[:, signal_len: ]), dim=1).squeeze(0)
            else:
                signal = signal.squeeze(0)
            return signal.squeeze(0).numpy()
        
    def cal_distortion_probs(self, distortion_probs):
        # append clean speech probability to distortion_probs
        clean_prob = 1 - sum(distortion_probs)
        assert clean_prob >= 0, "sum of distortion probabilities cannot exceed 1.0"
        return distortion_probs + [clean_prob]

    def add_multi_additive_distortions(self, signal, noise_types=None, sample_rate=None, snr=None):
        """_summary_
        Args:
            signal (numpy): the main signal, shape: [T]
            noise_types (list, optional): list of noise types. Defaults to None. If None, random pick.
            snr (float, optional): The desired snr. Defaults to None.
            sample_rate (int, optional): The desired sample rate. Defaults to None.
        """
        if noise_types == None:
            noise_num = random.randint(1,3)
            noise_types = np.random.choice(['mn', 'ms', 'mm', 'g', 'dns', 'wham', 'fsd'], size=noise_num, replace=False)
        
        noises = []
        for noise_type in noise_types:
            if noise_type == 'g':
                noise = np.random.randn(signal.shape[0])
                gau_snr = random.choice(range(10, 20, 1))
                coeff = self._snr_coeff(gau_snr, signal, noise)
                noise *= coeff
            elif noise_type == 'm':
                noise_path = random.choice(self.musan_files)
                noise = self.load_wav(noise_path, sample_rate)
            elif noise_type == 'mn':
                noise_path = random.choice(self.musan_noise_files)
                noise = self.load_wav(noise_path, sample_rate)
            elif noise_type == 'ms':
                noise_path = random.choice(self.musan_speech_files)
                noise = self.load_wav(noise_path, sample_rate)
            elif noise_type == 'mm':
                noise_path = random.choice(self.musan_music_files)
                noise = self.load_wav(noise_path, sample_rate)
            elif noise_type == 'wham':
                noise_path = random.choice(self.wham_noise_files)
                noise = self.load_wav(noise_path, sample_rate)
            elif noise_type == 'fsd':
                noise_path = random.choice(self.fsd50k_noise_files)
                noise = self.load_wav(noise_path, sample_rate)
            elif noise_type == 'dns':
                noise_path = random.choice(self.dns_noise_files)
                noise = self.load_wav(noise_path, sample_rate)
            
            signal_len = signal.shape[0]
            noise_len = noise.shape[0]
            if signal_len <= noise_len:
                start = random.randint(0, noise_len - signal_len)
                noise = noise[start: start+signal_len]
            else:
                n_repeat = signal_len // noise_len + 1
                noise = np.repeat(noise, n_repeat)
                noise = noise[: signal_len]
                
            noises.append(noise)
        
        noises = np.stack(noises, axis = 0)
        snr = random.choice(range(10, 20, 1)) if snr is None else snr
        c = np.random.rand(len(noise_types), 1)
        _noises = c * noises
        _noises = np.sum(_noises, axis=0)
        pwr_signal = sum(signal ** 2)
        pwr_noise = sum(_noises ** 2)
        Y = (pwr_signal * 10 ** (-snr / 10)) ** 0.5
        X = pwr_noise ** 0.5
        c_snr = c * Y / X
        noises = c_snr * noises
        noises = np.sum(noises, axis=0)
        pwr_noises = sum(noises ** 2)
        # assert 10 * log10(pwr_signal/pwr_noises) == snr, f"we want snr = {snr}, but we get {10 * log10(pwr_signal/pwr_noises)}"
        return signal + noises
    
    def add_multi_distortions(self, signal, add_dist_type, non_add_dist_type, sample_rate=None, snr=None):
        """_summary_

        Args:
            signal (numpy): the main signal, shape: [T]
            noise_types (list, optional): list of noise types. Defaults to None. If None, random pick.
            snr (float, optional): The desired snr. Defaults to None.
            sample_rate (int, optional): The desired sample rate. Defaults to None.
        """
            
        if add_dist_type == 'g':
            noise = np.random.randn(signal.shape[0])
            gau_snr = random.choice(range(10, 20, 1))
            coeff = self._snr_coeff(gau_snr, signal, noise)
            noise *= coeff
        elif add_dist_type == 'm':
            noise_path = random.choice(self.musan_files)
            noise = self.load_wav(noise_path, sample_rate)
        elif add_dist_type == 'mn':
            noise_path = random.choice(self.musan_noise_files)
            noise = self.load_wav(noise_path, sample_rate)
        elif add_dist_type == 'ms':
            noise_path = random.choice(self.musan_speech_files)
            noise = self.load_wav(noise_path, sample_rate)
        elif add_dist_type == 'mm':
            noise_path = random.choice(self.musan_music_files)
            noise = self.load_wav(noise_path, sample_rate)
        elif add_dist_type == 'wham':
            noise_path = random.choice(self.wham_noise_files)
            noise = self.load_wav(noise_path, sample_rate)
        elif add_dist_type == 'fsd':
            noise_path = random.choice(self.fsd50k_noise_files)
            noise = self.load_wav(noise_path, sample_rate)
        elif add_dist_type == 'dns':
            noise_path = random.choice(self.dns_noise_files)
            noise = self.load_wav(noise_path, sample_rate)
            rir_path = random.choice(self.dns_rir_files)
            rir = self.load_wav(rir_path, sample_rate)
        elif add_dist_type == 'c':
            noise = np.zeros(signal.shape[0])
        else:
            raise ValueError(f"We do not support add_dist_type like {add_dist_type}")
            
        signal_len = signal.shape[0]
        noise_len = noise.shape[0]
        if signal_len <= noise_len:
            start = random.randint(0, noise_len - signal_len)
            noise = noise[start: start+signal_len]
        else:
            n_repeat = signal_len // noise_len + 1
            noise = np.repeat(noise, n_repeat)
            noise = noise[: signal_len]
        
        snr = random.choice(range(10, 20, 1)) if snr is None else snr
        if sum(abs(noise)) != 0:
            coeff = self._snr_coeff(snr, signal, noise)
            noise *= coeff
            signal += noise
        
        if add_dist_type == 'dns':
            rir_snr = random.choice(range(10, 20, 1))
            signal = self.add_real_rir(signal, rir, rir_snr)
        
        chain = augment.EffectChain()
        if non_add_dist_type == 'b':
            chain = chain.sinc('-a', '120', SpecAugmentBand(sample_rate, self.distortion_cfg["bandrej"]["band_scaler"]))
        elif non_add_dist_type == 'p':
            pitch_randomizer = RandomPitchShift(self.distortion_cfg["pitch"]["pitch_shift_max"])
            if self.distortion_cfg["pitch"]["pitch_quick"]:
                chain = chain.pitch('-q', pitch_randomizer).rate('-q', sample_rate)
            else:
                chain = chain.pitch(pitch_randomizer).rate(sample_rate)
        elif non_add_dist_type == 'r':
            randomized_params = RandomReverb(self.distortion_cfg["reverb"]["reverberance_min"], self.distortion_cfg["reverb"]["reverberance_max"], 
                                self.distortion_cfg["reverb"]["damping_min"], self.distortion_cfg["reverb"]["damping_max"], self.distortion_cfg["reverb"]["room_scale_min"], self.distortion_cfg["reverb"]["room_scale_max"])
            chain = chain.reverb(randomized_params).channels()
        elif non_add_dist_type == 't':
            chain = chain.time_dropout(max_seconds=self.distortion_cfg["time_drop"]["t_ms"] / 1000.0)
        elif non_add_dist_type == 'clip':
            chain = chain.clip(RandomClipFactor(self.distortion_cfg["clip"]["clip_min"], self.distortion_cfg["clip"]["clip_max"])) 
        elif non_add_dist_type == 'c':
            chain = None
        else:
            raise ValueError(f"We do not support non_add_dist_type like {non_add_dist_type}")
        
        ori_signal_len = len(signal)    
        ori_signal = torch.Tensor(signal).unsqueeze(0)
        
        if chain != None:
            signal = chain.apply(ori_signal, 
                    src_info=dict(rate=sample_rate, length=ori_signal.size(1), channels=ori_signal.size(0)),
                    target_info=dict(rate=sample_rate, length=0)
            )
            signal_len = signal.size(1)
            # deal with inconsistent length of distorted and original speech
            if signal_len > ori_signal_len:
                signal = signal[:, :ori_signal_len].squeeze(0)
            elif signal_len < ori_signal_len:
                signal = torch.cat((signal, ori_signal[:, signal_len: ]), dim=1).squeeze(0)
            else:
                signal = signal.squeeze(0)
        else:
            signal = ori_signal.squeeze(0)
        
        assert len(signal) == ori_signal_len
        return signal.numpy()

if __name__ == '__main__':
    distortion_types = ['ms', 'mm', 'mn', 'g', 'r', 'dns', 'fsd', 'wham'] + ['c'] # m: musan noise, g: Gaussian, r: reverberation, c: clean
    distortion_probs = [0.3, 0.4, 0.3]
    distortion_config = './distortion_config.yaml'

    with open(distortion_config, 'r') as f:
        distortion_cfg = yaml.load(f, Loader=yaml.CLoader)
        f.close()
    DF = DistortionFactory(distortion_types=distortion_types, distortion_cfg=distortion_cfg)
    # distortion_probs = DF.cal_distortion_probs() # should be [0.3, 0.4, 0.3, 0.0]
    # choice = np.random.choice(distortion_types, 1, p=distortion_probs)[0]
    wav, sr = torchaudio.load('/work/twsgzqx489/corpus/fluent_speech_commands_dataset/wavs/speakers/2BqVo8kVB2Skwgyb/029f6450-447a-11e9-a9a5-5dbec3b8816a.wav')    
    wav = wav.squeeze(0)
    # if not choice == 'c':
    #     wav = self.DF.add_distortion(wav.numpy(), choice, sr)
    
    # wav = DF.add_multi_additive_distortions(wav.numpy(), ['ms', 'mm', 'g', 'd', 'w'], sr)
    # wav = DF.add_multi_distortions(wav.numpy(), ['mn', 'ms', 'mm', 'g', 'wham', 'c'], ['r', 'p', 'b', 'clip', 'c'], sr, 20)
    distorted_wav = DF.add_multi_distortions(wav.numpy(), ['g'], ['c'], sr, 10)
    sf.write('./test_augment.wav', distorted_wav, sr)
