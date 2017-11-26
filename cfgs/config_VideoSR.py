from easydict import EasyDict as edict

cfg = edict()

# Motion Estimation module
cfg.motion_estimation = edict()
cfg.motion_estimation.H = 100
cfg.motion_estimation.W = 100
cfg.motion_estimation.coarse_flow = edict()
cfg.motion_estimation.coarse_flow.k_size = [5, 3, 5, 3, 3]
cfg.motion_estimation.coarse_flow.ch_out = [24, 24, 24, 24, 32]
cfg.motion_estimation.coarse_flow.stide = [2, 1, 2, 1, 1]
cfg.motion_estimation.fine_flow = edict()
cfg.motion_estimation.fine_flow.k_size = [5, 3, 3, 3, 3]
cfg.motion_estimation.fine_flow.ch_out = [24, 24, 24, 24, 8]
cfg.motion_estimation.fine_flow.stide = [2, 1, 1, 1, 1]

# SPMC layer
cfg.upscale_factor = 4

# Detail Fusion Net
cfg.detail_fusion_net = edict()
cfg.detail_fusion_net.encoder = edict()
cfg.detail_fusion_net.encoder.k_size = [5, 3, 3, 3]
cfg.detail_fusion_net.encoder.ch_out = [32, 64, 64, 128]
cfg.detail_fusion_net.encoder.stide = [1, 2, 1, 2]

cfg.detail_fusion_net.decoder = edict()
cfg.detail_fusion_net.decoder.k_size = [3, 4, 3, 4, 3, 5]
cfg.detail_fusion_net.decoder.ch_out = [128, 64, 64, 32, 32, 1]
cfg.detail_fusion_net.decoder.stide = [1, 2, 1, 2, 1, 1]
cfg.detail_fusion_net.decoder.type = ['conv', 'deconv', 'conv', 'deconv', 'conv', 'conv']




