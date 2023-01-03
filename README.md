# distort-robust-distilSSL

This repository is based on [s3prl](https://github.com/s3prl/s3prl), and is not an official release or development of s3prl.
- Code for [Improving Generalizability of Distilled Self-supervised Speech Processing Models under Distorted Settings](https://arxiv.org/abs/2210.07978).
- Currently supports five downstream tasks: `IC, ER, KS, ASR`

## How to run
- Modify the `superb` folder name into `s3prl`. Install the official [s3prl](https://github.com/s3prl/s3prl) toolkit according to the installation guidelines. Replace or move the files in this repository to the official [s3prl](https://github.com/s3prl/s3prl) repository following to the corresponding file structure.

- `-o` or `--override` can override any argument or config field with command line, which is at the highest priority. Please refer to the [override function](https://github.com/s3prl/s3prl/blob/master/s3prl/utility/helper.py) for definition.
- Support distortion modes:
  - None
  - single
  - double
- Support distortion types: 
  - usage ex: `config.downstream_expert.datarc.distortion_types=['m','g','r']`
    - m: Musan noise
    - g: Gaussian noise
    - r: Reverberation
    - b: band rejection
    - p: pitch shift
    - fsd: FSD50k (only used for testing)
    - dns: DNS (only used for testing)

### Example

- setup1 distillation (teacher clean, student distorted)
```bash
python3 run_pretrain.py -u distiller \
-g pretrain/distiller/config_model.yaml \
-n setup1_2-dis_t_nocont \
-o "\
config.pretrain_expert.datarc.teacher.distortion_same_as_student=False,,\
config.pretrain_expert.datarc.teacher.distortion_mode=None,,\
config.pretrain_expert.datarc.student.distortion_mode=double\
"
```

- setup2 distillation (teacher distorted, student distorted, different distortions)
```bash
python3 run_pretrain.py -u distiller \
-g pretrain/distiller/config_model.yaml \
-n setup2_2-dis_t_nocont \
-o "\
config.pretrain_expert.datarc.teacher.distortion_same_as_student=False,,\
config.pretrain_expert.datarc.teacher.distortion_mode=double,,\
config.pretrain_expert.datarc.student.distortion_mode=double\
"
```

- IC downstream fine-tuning
```bash
python3 run_downstream.py \
-m train \
-d fluent_commands \
-u distiller_local \
-k result/pretrain/setup1_2-dis_t_nocont/dev-dis-best.ckpt \
-n ic_paper/setup1_2-dis_t_nocont \
-s paper \
-o "\
config.downstream_expert.datarc.distortion_mode=None
"
```

- IC downstream testing for 2-dis
```bash
python3 run_downstream.py \
-m evaluate \
-e result/downstream/ic_paper/setup1_2-dis_t_nocont/dev-best.ckpt \
-o "\
config.downstream_expert.datarc.distortion_mode=double,,\
config.downstream_expert.datarc.additive_dist_types=['m', 'g', 'wham'],,\
config.downstream_expert.datarc.non_additive_dist_types=['r', 'p', 'b']\
"
```

- IC downstream testing for DNS or FSD50k
```bash
python3 run_downstream.py \
-m evaluate \
-e result/downstream/ic_paper/setup1_2-dis_t_nocont/dev-best.ckpt \
-o "\
config.downstream_expert.datarc.distortion_mode=single,,\
config.downstream_expert.datarc.distortion_types=['dns']
```


## Citation
If you find this code useful, please consider citing following papers.
```
@inproceedings{huang22b_interspeech,
  author={Kuan Po Huang and Yu-Kuan Fu and Yu Zhang and Hung-yi Lee},
  title={{Improving Distortion Robustness of Self-supervised Speech Processing Tasks with Domain Adaptation}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={2193--2197},
  doi={10.21437/Interspeech.2022-519}
}

@inproceedings{RobustDistilHuBERT,
  title={Improving generalizability of distilled self-supervised speech processing models under distorted settings},
  author={Huang, Kuan-Po and
  Fu, Yu-Kuan and
  Hsu, Tsu-Yuan and
  Fabian Ritter Gutierrez and
  Wang, Fan-Lin and
  Tseng, Liang-Hsuan and
  Zhang, Yu and
  others},
  booktitle={IEEE-SLT Workshop},
  year={2022}
}
```
