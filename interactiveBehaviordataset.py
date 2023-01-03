import torch
from torch.utils import data
import torchvision.models as models
import torchvision.transforms as transforms

import random
from PIL import Image
import numpy as np

from collections import Counter


FRAMES_NUM={1: 148, 2: 328, 3: 39, 4: 251, 5: 38, 6: 208, 7: 16, 8: 21, 9: 18, 10: 17, 
            11: 152, 12: 213, 13: 30, 14: 17, 15: 39, 16: 60, 17: 15, 18: 104, 19: 63, 20: 101, 
            21: 552, 22: 38, 23: 9, 24: 58, 25: 13, 26: 45, 27: 57, 28: 32, 29: 641, 30: 27, 
            31: 22, 32: 9, 33: 544, 34: 68, 35: 574, 36: 61, 37: 26, 38: 8, 39: 81, 40: 105, 
            41: 269, 42: 42, 43: 389, 44: 10, 45: 97, 46: 135, 47: 70, 48: 96, 49: 946, 50: 78, 
            51: 123, 52: 19, 53: 171, 54: 13, 55: 17, 56: 269, 57: 29, 58: 712, 59: 58, 60: 14, 
            61: 73, 62: 15, 63: 741, 64: 63, 65: 17, 66: 19, 67: 399, 68: 20, 69: 374, 70: 45, 
            71: 180, 72: 119, 73: 84, 74: 64, 75: 32, 76: 82, 77: 25, 78: 445, 79: 4, 80: 176, 
            81: 51, 82: 41, 83: 12, 84: 100, 85: 80, 86: 259, 87: 12, 88: 19, 89: 43, 90: 334, 
            91: 46, 92: 65, 93: 25, 94: 46, 95: 43, 96: 18, 97: 26, 98: 40, 99: 20, 100: 598}
 
FRAMES_SIZE={1:(1080, 1920), 2:(1080, 1920), 3:(1080, 1920), 4:(1080, 1920), 5:(1080, 1920), 6:(1080, 1920), 7:(1080, 1920), 8:(1080, 1920), 9:(1080, 1920), 10:(1080, 1920), 
             11: (1080, 1920), 12: (1080, 1920), 13:(1080, 1920), 14:(1080, 1920), 15:(1080, 1920), 16:(1080, 1920), 17:(1080, 1920), 18:(1080, 1920), 19:(1080, 1920), 20:(1080, 1920), 
             21: (1080, 1920), 22: (1080, 1920), 23: (1080, 1920), 24: (1080, 1920), 25:(1080, 1920), 26:(1080, 1920), 27:(1080, 1920), 28:(1080, 1920), 29:(1080, 1920), 30:(1080, 1920), 
             31: (1080, 1920), 32: (1080, 1920), 33: (1080, 1920), 34:(1080, 1920), 35:(1080, 1920), 36:(1080, 1920), 37:(1080, 1920), 38:(1080, 1920), 39:(1080, 1920), 40:(1080, 1920), 
             41: (1080, 1920), 42: (1080, 1920), 43: (1080, 1920), 44: (1080, 1920), 45:(1080, 1920), 46:(1080, 1920), 47:(1080, 1920), 48:(1080, 1920), 49:(1080, 1920), 50:(1080, 1920), 
             51: (1080, 1920), 52: (1080, 1920), 53: (1080, 1920), 54: (1080, 1920), 55: (1080, 1920), 56:(1080, 1920), 57:(1080, 1920), 58:(1080, 1920), 59:(1080, 1920), 60:(1080, 1920),
             61: (1080, 1920), 62: (1080, 1920), 63: (1080, 1920), 64: (1080, 1920), 65:(1080, 1920), 66:(1080, 1920), 67:(1080, 1920), 68:(1080, 1920), 69:(1080, 1920), 70:(1080, 1920), 
             71: (1080, 1920), 72: (1080, 1920), 73: (1080, 1920), 74: (1080, 1920), 75:(1080, 1920), 76:(1080, 1920), 77:(1080, 1920), 78:(1080, 1920), 79:(1080, 1920), 80:(1080, 1920), 
             81: (1080, 1920), 82: (1080, 1920), 83: (1080, 1920), 84: (1080, 1920), 85:(1080, 1920), 86:(1080, 1920), 87:(1080, 1920), 88:(1080, 1920), 89:(1080, 1920), 90:(1080, 1920), 
             91: (1080, 1920), 92: (1080, 1920), 93: (1080, 1920), 94: (1080, 1920), 95:(1080, 1920), 96:(1080, 1920), 97:(1080, 1920), 98:(1080, 1920), 99:(1080, 1920), 100:(1080, 1920)}


ACTIONS=['accept','deny','asking','talking','Conversation','answering','doing-homework','teaching','ordering']
ACTIVITIES=['Teaching-listening','Ordering-executing','Discussion in teaching','Talking in group','student-student','teacher-student','teacher-teacher']

ACTIONS_ID={a:i for i,a in enumerate(ACTIONS)}
ACTIVITIES_ID={a:i for i,a in enumerate(ACTIVITIES)}


def interaction_read_annotations(path,sid):
    annotations={}
    path=path + '/seq%03d/annotations.txt' % sid
    
    with open(path,mode='r') as f:
        frame_id=None
        group_activity=None
        actions=[]
        bboxes=[]
        group_activity_total=[]
        for l in f.readlines():
            #values=l[:-1].split('	')
            values=l[:-1].split(' ')
            #if (int(values[1])>200 and int(values[2])>292 and int(values[3])<504 and int(values[4])<444) or (int(values[1])>1194 and int(values[2])>228 and int(values[3])<1300 and int(values[4])<326):
            if (int(values[1])>860 and int(values[2])>156 and int(values[3])<1318 and int(values[4])<340):
                continue
            else:                
                if int(values[0])!=frame_id:
                    if frame_id!=None and frame_id%10==1 and frame_id+9<=FRAMES_NUM[sid]:
                        counter = Counter(group_activity_total).most_common(2)#Counter(actions).most_common(2)
                        #print(counter)
                        #print(counter[0][0])
                        #print(counter[1][0])
                        #就选最多的活动
                        group_activity= counter[0][0]#counter[0][0]-1 if counter[0][0]!=0 else counter[1][0]-1
                        annotations[frame_id]={
                            'frame_id':frame_id,
                            'group_activity':group_activity,
                            'actions':actions,
                            'bboxes':bboxes
                        }

                    frame_id=int(values[0])
                    group_activity=None
                    actions=[]
                    bboxes=[]
                    group_activity_total=[]

                #活动的编号修改
                item = int(values[6])
                group_activity_total.append(item-1 if item<5 else item-47)
                #print(group_activity_total)
                actions.append(int(values[5])-1)
                x,y,w,h = (int(values[i])  for i  in range(1,5))
                H,W=FRAMES_SIZE[sid]

                bboxes.append( (y/H,x/W,(y+h)/H,(x+w)/W) )
            '''
            if frame_id!=None and frame_id%10==1 and frame_id+9<=FRAMES_NUM[sid]:
                counter = Counter(actions).most_common(2)
                group_activity= counter[0][0]-1 if counter[0][0]!=0 else counter[1][0]-1
                annotations[frame_id]={
                    'frame_id':frame_id,
                    'group_activity':group_activity,
                    'actions':actions,
                    'bboxes':bboxes
                }
            '''

    return annotations
            
        
        
def interaction_read_dataset(path,seqs):
    data = {}
    for sid in seqs:
        data[sid] = interaction_read_annotations(path,sid)
    return data

def interaction_all_frames(anns):
    return [(s,f)  for s in anns for f in anns[s] ]


class InteractionDataset(data.Dataset):
    """
    Characterize collective dataset for pytorch
    """
    def __init__(self,anns,frames,images_path,image_size,feature_size,num_boxes=13,num_frames=10,is_training=True,is_finetune=False):
        self.anns=anns
        self.frames=frames
        self.images_path=images_path
        self.image_size=image_size
        self.feature_size=feature_size
        
        self.num_boxes=num_boxes
        self.num_frames=num_frames
        
        self.is_training=is_training
        self.is_finetune=is_finetune
    
    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.frames)
    
    def __getitem__(self,index):
        """
        Generate one sample of the dataset
        """
        
        select_frames=self.get_frames(self.frames[index])
        
        sample=self.load_samples_sequence(select_frames)
        
        return sample
    
    def get_frames(self,frame):
        
        sid, src_fid = frame
        
        if self.is_finetune:
            if self.is_training:
                fid=random.randint(src_fid, src_fid+self.num_frames-1)
                return [(sid, src_fid, fid)]
        
            else:
                return [(sid, src_fid, fid) 
                        for fid in range(src_fid, src_fid+self.num_frames)]
            
        else:
            if self.is_training:
                sample_frames=random.sample(range(src_fid,src_fid+self.num_frames),3)
                return [(sid, src_fid, fid) for fid in sample_frames]

            else:
                sample_frames=[ src_fid, src_fid+3, src_fid+6, src_fid+1, src_fid+4, src_fid+7, src_fid+2, src_fid+5, src_fid+8 ]
                return [(sid, src_fid, fid) for fid in sample_frames]
    
    
    def load_samples_sequence(self,select_frames):
        """
        load samples sequence

        Returns:
            pytorch tensors
        """
        OH, OW=self.feature_size
        
        images, bboxes = [], []
        activities, actions = [], []
        bboxes_num=[]
    
        
        for i, (sid, src_fid, fid) in enumerate(select_frames):

            img = Image.open(self.images_path + '/seq%03d/Image%d.jpg'%(sid,fid))

            img=transforms.functional.resize(img,self.image_size)
            img=np.array(img)

            # H,W,3 -> 3,H,W
            img=img.transpose(2,0,1)
            images.append(img)
            
            temp_boxes=[]
            for box in self.anns[sid][src_fid]['bboxes']:
                y1,x1,y2,x2=box
                w1,h1,w2,h2 = x1*OW, y1*OH, x2*OW, y2*OH  
                temp_boxes.append((w1,h1,w2,h2))
                
            temp_actions=self.anns[sid][src_fid]['actions'][:]
            bboxes_num.append(len(temp_boxes))
            
            while len(temp_boxes)!=self.num_boxes:
                temp_boxes.append((0,0,0,0))
                temp_actions.append(-1)
            
            bboxes.append(temp_boxes)
            actions.append(temp_actions)
            
            activities.append(self.anns[sid][src_fid]['group_activity'])
        
        
        images = np.stack(images)
        activities = np.array(activities, dtype=np.int32)
        bboxes_num = np.array(bboxes_num, dtype=np.int32)
        bboxes=np.array(bboxes,dtype=np.float).reshape(-1,self.num_boxes,4)
        actions=np.array(actions,dtype=np.int32).reshape(-1,self.num_boxes)
        
        #convert to pytorch tensor
        images=torch.from_numpy(images).float()
        bboxes=torch.from_numpy(bboxes).float()
        actions=torch.from_numpy(actions).long()
        activities=torch.from_numpy(activities).long()
        bboxes_num=torch.from_numpy(bboxes_num).int()
        
        return images, bboxes,  actions, activities, bboxes_num
    
    

    
