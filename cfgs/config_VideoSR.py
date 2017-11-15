from easydict import EasyDict as edict

cfg = edict()

cfg.coarse_flow = edict()
cfg.coarse_flow.k_size = [5, 3, 5, 3, 3]
cfg.coarse_flow.ch_out = [24, 24, 24, 24, 32]
cfg.coarse_flow.stide = [2, 1, 2, 1, 1]

cfg.fine_flow = edict()
cfg.fine_flow.k_size = [5, 3, 3, 3, 3]
cfg.fine_flow.ch_out = [24, 24, 24, 24, 8]
cfg.fine_flow.stide = [2, 1, 1, 1, 1]