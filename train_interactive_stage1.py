import sys
sys.path.append(".")
from train_net import *

cfg=Config('interactiveBehaviordataset')

cfg.device_list="0,1"
cfg.training_stage=1
cfg.train_backbone=True

cfg.image_size=1080, 1920
cfg.out_size=57,87
cfg.num_boxes=35
cfg.num_actions=9
cfg.num_activities=7
cfg.num_frames=10

cfg.batch_size=2
cfg.test_batch_size=1
cfg.train_learning_rate=1e-5
cfg.train_dropout_prob=0.5
cfg.weight_decay=1e-2
cfg.lr_plan={}
cfg.max_epoch=100

cfg.exp_note='Interaction_stage1'
train_net(cfg)