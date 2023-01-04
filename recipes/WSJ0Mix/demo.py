#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :demo.py
# @Time      :2023/1/4 下午2:39
# @Author    :Aliang


from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

model = separator.from_hparams(source="speechbrain/sepformer-wsj03mix", savedir='pretrained_models/sepformer-wsj03mix')

# for custom file, change path
est_sources = model.separate_file(path='/home/stary/code/dataset/spatialize_wsj0-mix/405o0319_2.3824_01xo030w_-2.3824.wav')

torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)
