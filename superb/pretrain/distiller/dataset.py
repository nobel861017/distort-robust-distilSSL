"""
    Dataset for distiller
    Author: Kuan Po, Huang
"""

import os
import yaml
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from pretrain.bucket_dataset import WaveDataset
from pretrain.distortions import DistortionFactory

np.random.seed(7122)

class OnlineWaveDataset(WaveDataset):
    """Online waveform dataset"""

    def __init__(
        self,
        task_config,
        bucket_size,
        file_path,
        sets,
        max_timestep=0,
        libri_root=None,
        target_level=-25,
        data_type=None,
        **kwargs
    ):
        super().__init__(
            task_config,
            bucket_size,
            file_path,
            sets,
            max_timestep,
            libri_root,
            **kwargs
        )
        self.target_level = target_level

        self.teacher_distortion_same_as_student = kwargs['teacher']['distortion_same_as_student']
        self.teacher_distortion_types = None
        self.teacher_distortion_mode = kwargs['teacher']['distortion_mode']
        self.student_distortion_mode = kwargs['student']['distortion_mode']
        if not self.teacher_distortion_same_as_student:
            if self.teacher_distortion_mode == 'single':
                self.teacher_distortion_types = kwargs['teacher']["distortion_types"] + ['c'] if kwargs['teacher']["distortion_types"] is not None else None # 'c' for clean
                if self.teacher_distortion_types is not None:
                    self.teacher_distortion_config = kwargs['teacher']["distortion_config"]
                    self.teacher_DF, self.teacher_distortion_probs = self.build_DF_and_distortion_probs_for_single_mode(self.teacher_distortion_types, self.teacher_distortion_config, data_type)
                else:
                    ValueError(f"distortion_mode single requires specification of distortion_types")
            elif self.teacher_distortion_mode == 'double':
                self.teacher_additive_dist_types = kwargs['teacher']["additive_dist_types"] + ['c'] if kwargs['teacher']["additive_dist_types"] is not None else None # 'c' for clean
                self.teacher_non_additive_dist_types = kwargs['teacher']["non_additive_dist_types"] + ['c'] if kwargs['teacher']["non_additive_dist_types"] is not None else None # 'c' for clean
                if (self.teacher_additive_dist_types is not None) and (self.teacher_non_additive_dist_types is not None):
                    self.teacher_distortion_config = kwargs['teacher']["distortion_config"]
                    self.teacher_DF = self.build_DF_and_distortion_probs_for_double_mode(self.teacher_additive_dist_types + self.teacher_non_additive_dist_types, self.teacher_distortion_config, data_type)
            else:
                print('Not adding distortions to teacher input.')
        if self.student_distortion_mode == 'single':
            self.student_distortion_types = kwargs['student']["distortion_types"] + ['c'] if kwargs['student']["distortion_types"] is not None else None # 'c' for clean
            if self.student_distortion_types is not None:
                self.student_distortion_config = kwargs['student']["distortion_config"]
                self.student_DF, self.student_distortion_probs = self.build_DF_and_distortion_probs_for_single_mode(self.student_distortion_types, self.student_distortion_config, data_type)
            else:
                ValueError(f"distortion_mode single requires specification of distortion_types")
        elif self.student_distortion_mode == 'double':
            self.student_additive_dist_types = kwargs['student']["additive_dist_types"] + ['c'] if kwargs['student']["additive_dist_types"] is not None else None # 'c' for clean
            self.student_non_additive_dist_types = kwargs['student']["non_additive_dist_types"] + ['c'] if kwargs['student']["non_additive_dist_types"] is not None else None # 'c' for clean
            if (self.student_additive_dist_types is not None) and (self.student_non_additive_dist_types is not None):
                self.student_distortion_config = kwargs['student']["distortion_config"]
                self.student_DF = self.build_DF_and_distortion_probs_for_double_mode(self.student_additive_dist_types + self.student_non_additive_dist_types, self.student_distortion_config, data_type)
        else:
            print('Not adding distortions to student input.')

    def build_DF_and_distortion_probs_for_single_mode(self, distortion_types, distortion_cfg, data_type):
        
        with open(distortion_cfg, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.CLoader)
            f.close()
        DF = DistortionFactory(distortion_types, cfg, data_type)
        if cfg["distortion_probs"] is not None:
            distortion_probs = DF.cal_distortion_probs(cfg["distortion_probs"])
        else:
            distortion_probs = [1.0 / (len(distortion_types))]*len(distortion_types)
        assert len(distortion_probs) == len(distortion_types)
        print(distortion_probs, distortion_types)
        return DF, distortion_probs

    def build_DF_and_distortion_probs_for_double_mode(self, distortion_types, distortion_cfg, data_type):
        with open(distortion_cfg, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.CLoader)
            f.close()
        DF = DistortionFactory(distortion_types, cfg, data_type)
        print(distortion_types)
        return DF

    def _load_feat(self, feat_path):
        if self.libri_root is None:
            return torch.FloatTensor(np.load(os.path.join(self.root, feat_path)))
        
        student_wav, sr = torchaudio.load(os.path.join(self.libri_root, feat_path))
        student_wav = student_wav.squeeze(0)
        teacher_wav = student_wav.clone()
        if (self.teacher_distortion_mode is not None) and (not self.teacher_distortion_same_as_student):
            if self.teacher_distortion_mode == 'single':
                t_choice = np.random.choice(self.teacher_distortion_types, 1, p=self.teacher_distortion_probs)[0]
                if not t_choice == 'c':
                    teacher_wav = self.teacher_DF.add_distortion(teacher_wav.numpy(), t_choice, sr)
                t_choice = [t_choice]
            elif self.teacher_distortion_mode == 'double':
                add_dist_type = np.random.choice(self.teacher_additive_dist_types)
                non_add_dist_type = np.random.choice(self.teacher_non_additive_dist_types)
                teacher_wav = self.teacher_DF.add_multi_distortions(
                    teacher_wav.numpy(), 
                    add_dist_type=add_dist_type, 
                    non_add_dist_type=non_add_dist_type, 
                    sample_rate=sr
                )
                t_choice = [add_dist_type, non_add_dist_type]
            if type(teacher_wav) is not torch.Tensor:
                teacher_wav = torch.from_numpy(teacher_wav)
        else:
            t_choice = ['c']
            
        if self.student_distortion_mode is not None:
            if self.student_distortion_mode == 'single':
                s_choice = np.random.choice(self.student_distortion_types, 1, p=self.student_distortion_probs)[0]
                if not s_choice == 'c':
                    student_wav = self.student_DF.add_distortion(student_wav.numpy(), s_choice, sr)
                s_choice = [s_choice]
            elif self.student_distortion_mode == 'double':
                add_dist_type = np.random.choice(self.student_additive_dist_types)
                non_add_dist_type = np.random.choice(self.student_non_additive_dist_types)
                student_wav = self.student_DF.add_multi_distortions(
                    student_wav.numpy(), 
                    add_dist_type=add_dist_type, 
                    non_add_dist_type=non_add_dist_type, 
                    sample_rate=sr
                )
                s_choice = [add_dist_type, non_add_dist_type]
            if type(student_wav) is not torch.Tensor:
                student_wav = torch.from_numpy(student_wav)
        else:
            s_choice = ['c']
        
        if self.teacher_distortion_same_as_student:
            return student_wav.squeeze(), student_wav.squeeze()
        else:
            return teacher_wav.squeeze(), student_wav.squeeze()

    def __getitem__(self, index):
        # Load acoustic feature and pad
        x_batch = [self._sample(self._load_feat(x_file)) for x_file in self.X[index]]
        teacher_x_batch, student_x_batch = [x[0] for x in x_batch], [x[1] for x in x_batch]

        teacher_x_lens = [len(x) for x in teacher_x_batch]
        teacher_x_lens = torch.LongTensor(teacher_x_lens)
        teacher_x_pad_batch = pad_sequence(teacher_x_batch, batch_first=True)
        teacher_pad_mask = torch.ones(teacher_x_pad_batch.shape)  # (batch_size, seq_len)
        # zero vectors for padding dimension
        for idx in range(teacher_x_pad_batch.shape[0]):
            teacher_pad_mask[idx, teacher_x_lens[idx] :] = 0

        student_x_lens = [len(x) for x in student_x_batch]
        student_x_lens = torch.LongTensor(student_x_lens)
        student_x_pad_batch = pad_sequence(student_x_batch, batch_first=True)
        student_pad_mask = torch.ones(student_x_pad_batch.shape)  # (batch_size, seq_len)
        # zero vectors for padding dimension
        for idx in range(student_x_pad_batch.shape[0]):
            student_pad_mask[idx, student_x_lens[idx] :] = 0

        return [teacher_x_pad_batch, teacher_x_batch, teacher_x_lens, teacher_pad_mask], [student_x_pad_batch, student_x_batch, student_x_lens, student_pad_mask]
