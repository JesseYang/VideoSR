from easydict import EasyDict as edict

cfg = edict()

# model
cfg.frames = 5
cfg.h = 128
cfg.w = 128
cfg.upscale_factor = 4

# Motion Estimation module
cfg.motion_estimation = edict()
cfg.motion_estimation.H = 100
cfg.motion_estimation.W = 100
cfg.motion_estimation.coarse_flow = edict()
cfg.motion_estimation.coarse_flow.k_size = [5, 3, 5, 3, 3]
cfg.motion_estimation.coarse_flow.ch_out = [24, 24, 24, 24, 32]
cfg.motion_estimation.coarse_flow.stride = [2, 1, 2, 1, 1]
cfg.motion_estimation.fine_flow = edict()
cfg.motion_estimation.fine_flow.k_size = [5, 3, 3, 3, 3]
cfg.motion_estimation.fine_flow.ch_out = [24, 24, 24, 24, 8]
cfg.motion_estimation.fine_flow.stride = [2, 1, 1, 1, 1]

# Detail Fusion Net
cfg.detail_fusion_net = edict()
cfg.detail_fusion_net.encoder = edict()
cfg.detail_fusion_net.encoder.k_size = [5, 3, 3, 3]
cfg.detail_fusion_net.encoder.ch_out = [32, 64, 64, 128]
cfg.detail_fusion_net.encoder.stride = [1, 2, 1, 2]

cfg.detail_fusion_net.decoder = edict()
cfg.detail_fusion_net.decoder.k_size = [3, 4, 3, 4, 3, 5]
cfg.detail_fusion_net.decoder.ch_out = [128, 64, 64, 32, 32, 1]
cfg.detail_fusion_net.decoder.stride = [1, 2, 1, 2, 1, 1]
cfg.detail_fusion_net.decoder.type = ['conv', 'deconv', 'conv', 'deconv', 'conv', 'conv']

# Train
# Motion Estimation
cfg.lambda1 = 0.01
cfg.me_max_iteration = 70000
cfg.me_batch_size = 0
cfg.spmc_max_iteration = 20000

# data
cfg.train_list = ['data_train.txt']
cfg.test_list = 'data_test.txt'