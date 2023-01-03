import sys
sys.path.append(".")
from train_net import *

cfg=Config('interactiveBehaviordataset')

cfg.device_list="0,1"
cfg.training_stage=2
cfg.stage1_model_path='result/stage1_epoch30.pth'  #PATH OF THE BASE MODEL
cfg.train_backbone=False

cfg.image_size=1080, 1920
cfg.out_size=57,87
cfg.num_boxes=35
cfg.num_actions=9
cfg.num_activities=7
cfg.num_frames=10
cfg.num_graph=4
cfg.tau_sqrt=True

cfg.batch_size=2
cfg.test_batch_size=1 
cfg.train_learning_rate=1e-4
cfg.train_dropout_prob=0.2
cfg.weight_decay=1e-2
cfg.lr_plan={}
cfg.max_epoch=50

cfg.exp_note='Interaction_stage2'
train_net(cfg)