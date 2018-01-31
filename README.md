# VideoSR

# TODOs

- [ ] Motion Estimation
- [ ] SPMC layer
- [ ] Detail Fusion Net

# Usage

# File Tree
```python
VideoSR
├── cfgs                        # 数据集配置、模型配置
├── frame_seq_data              # 帧序列格式的数据放这里
├── video_data                  # 视频格式的数据放这里
├── modules                     # 网络模块
|   ├── __init__.py
|   ├── motion_estimation.py    # 基于CNN，详见VESPCN
|   ├── spmc_layer.py           # upscale + ForwardWarpping
|   └── detail_fusion_net.py    # Encoder + ConvLSTM + Decoder
├── train_log                   # Trained Models
|   ├── sintel_clean_me         # train ME on sintel clean
|   ├── sintel_clean_sr         # train SR on sintel clean (load from sintel_clean_me)
|   ├── sintel_clean_jointly    # train jointly on sintel clean (load from sintel_clean_jointly)
|
├── utils
|   ├── __init__.py
|   ├── flow.py                 # flow的I/O, 可视化等，Optical-Flow-Estimation中维护更新
|   ├── color.py                # 颜色空间转换相关 (TODO: 搞清楚几个函数的作用域和值域，似乎因为hr_y的范围已经影响到了模型训练)
|   ├── gradient.py             # FilteredGlobalNormClip的实现，论文中提到要对ConvLSTM的梯度单独作GlobalNormClip
|   └── warp.py                 # ForwardWarping的实现
├── reader.py                   # 读取帧序列，读取txt的格式要遵守与generate_txt.ipynb的约定
├── test.py                     # 单元测试的集合
├── train_new.py                # 包括了三个阶段训练的全部代码，目前通过stage来人工控制(TODO: 设计大数据下的数据集方案)(TODO: 第二、三阶段的训练)(TODO: 根据iterations自动改stage)
├── predict_new.py              # 预测flow、预测超分辨率结果、性能测试(TODO: 写stitch，提高通用性)(TODO: 读写方案?为每个数据集写helper?)(TODO: 测试新旧切图速度)(TODO: 需要写metrics, 查APE和EPE的定义)(TODO: 在Sintel上测试)
├── video_to_frames.py          # 视频转帧序列，对cap.read()的简单封装
├── config.yaml                 # 数据集路径配置，用于generate_txt.ipynb中
└── generate_txt.ipynb          # 从帧序列格式的数据集中生成txt，以供reader.py使用(TODO: 当前将连续帧的路径都存下来的方法，冗余大)
```
# Results

| Method        | Ours           | Paper  |
| :-------------: |:-------------:| :-----:|
| X2     | right-aligned | $1600 |
| X3      | centered      |   $12 |
| X4 | are neat      |    $1 |
# Trained Models
