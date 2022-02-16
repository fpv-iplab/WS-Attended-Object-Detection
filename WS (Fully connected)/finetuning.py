
import torch
import torch.nn as nn
import visdom
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import * 
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import copy
import os
import sklearn.metrics as metrics
from math import log2
from PIL import Image
from torch import autograd
import glob
import cv2
from sklearn import preprocessing
import random
import datetime
from scipy import sparse
import numpy as np
from scipy.sparse import csc_matrix
from torchvision.utils import save_image
from collections import Counter
import gzip
import PIL
import tensorflow as tf
from scipy.special import rel_entr, kl_div
import sys




def save_image_for_test(to_image ,pred_t):

  mapping_color_for_image={
    0: [230, 25, 75],
    1: [60, 180, 75],
    2: [205, 225, 25],
    3: [33, 60, 111],
    4: [245, 130, 48],
    5: [145, 30, 180],
    6: [0, 100, 240],
    7: [240, 50, 230],
    8: [210, 245, 60],
    9: [250, 190, 212],
    10:[0, 128, 128],
    11: [220, 190, 255],
    12: [170, 110, 40],
    13: [255, 250, 200],
    14: [128, 0, 0],
    15: [100,100,100],
  }

  current_array= [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
  for i in range(16):
    current_array[i] = np.ones((to_image.shape[0], to_image.shape[1], 3)) * mapping_color_for_image[i]
  dest = np.zeros((to_image.shape[0], to_image.shape[1], 3))

  for i in range(16):
    condition_array = np.zeros(to_image.shape)
    condition_array[to_image == i] = True
    condition_array = np.expand_dims(condition_array, axis=2)
    condition_array = np.repeat(condition_array, 3, axis=2)
    dest = np.where(condition_array, current_array[i], dest)

  print(pred_t)
  cv2.imwrite(pred_t.replace('.jpg','')+".png", dest) 
def get_gaze_patch(path,x,y,size,method): 
    img_clear = path

    x=int(x)
    y=int(y)
 
    half_size= int(size/2)
    min_y=y-half_size
    min_x=x-half_size
    max_y=y+half_size
    max_x=x+half_size
    if(method=='move_gaze'):
       
        if(y-size<0):
            y=y-(y-size) + 1
           
        if(x-size<0):
            x=x-(x-size)
            #print("x = 0 ")
        if(y+half_size>img_clear.shape[0]):
            #print("y > size ")
            y=img_clear.shape[0]#-half_size
        if(x+half_size>img_clear.shape[1]):
            #print("x > size ")
            x=img_clear.shape[1]#-half_size
        min_y=y-half_size
        min_x=x-half_size
        max_y=y+half_size
        max_x=x+half_size
        crop_img_small=    img_clear[min_y:max_y, min_x:max_x]
    else:
        min_y=y-half_size
        min_x=x-half_size
        max_y=y+half_size
        max_x=x+half_size
        position=""
        if(y-half_size<0):
            min_y = 1
            #max_y = y+(min_y+y)
           
            position= 'up'
        if(x-half_size<0):
            min_x=1
            #max_x = x+(min_x+x)
          
            position= 'left'
        if(y+half_size>img_clear.shape[0]):
            max_y=img_clear.shape[0]-1
            #min_y = y-(max_y-y)
            
            position= 'down'
        if(x+half_size>img_clear.shape[1]):
            max_x=img_clear.shape[1]-1
            #min_x = x-(max_x-x)
            
            position= 'right'
        crop_img_small = img_clear[min_y:max_y, min_x:max_x]
    
    if(method=='resize'):
      
        crop_img_small=    cv2.resize(img_clear[min_y:max_y, min_x:max_x], (size, size)) 

    if(method=='padding'):
       
        im1 =img_clear[min_y:max_y, min_x:max_x]
        shape = (size,size)
        h, w = im1.shape[0], im1.shape[1]
        color = [0, 0, 0]
        top, bottom, left, right = 0, 0, 0, 0
        new_im = im1.copy()
        diff = w - h
        
        if diff > 0:
            if diff%2 == 0: top, bottom = diff/2, diff/2
            else: top, bottom = diff/2+1, diff/2
            new_im = cv2.copyMakeBorder(new_im, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,value=color)

        elif diff < 0:
            diff = -diff
            if diff%2 == 0: left, right = diff/2, diff/2
            else: left, right = diff/2+1, diff/2
            new_im = cv2.copyMakeBorder(new_im, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,value=color)
            
        new_im = cv2.resize(new_im, shape, interpolation = cv2.INTER_AREA)
        return new_im
        #img_clear[0:max_y, min_x:max_x]#padding(Image.fromarray(img_clear[0:max_y, min_x:max_x]),(size))

    crop_img_small=np.array(crop_img_small)
    #cv2.circle(crop_img_small, ( int(crop_img_small.shape[1]/2), int(crop_img_small.shape[0]/2) ), 1, (0,255,0), 10)
    return(crop_img_small)

transforms_normal = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
transforms_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    #La rotation viene fatta dopo la patch
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
transforms_augmentation_2 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.GaussianBlur((11, 11), (10.5, 10.5)),
    #La rotation viene fatta dopo la patch
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
transforms_augmentation_3 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.GaussianBlur((23, 23), (10.5, 15.5)),
    #La rotation viene fatta dopo la patch
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

class GazeDataset_resized(Dataset):
    def __init__(self, images, labels,gaze_x,gaze_y ,transforms=None):
        self.X = images
        self.y = labels
        self.gaze_x = gaze_x
        self.gaze_y = gaze_y
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        
        
        img = cv2.imread(self.X[i])
        #img = get_gaze_patch(img, self.gaze_x[i], self.gaze_y[i],SIZE,METHOD)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       
        if self.transforms is not None:
            image_from_array = Image.fromarray(img)
            image = self.transforms(image_from_array).numpy()
            
        return image,self.y[i],self.X[i]


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def evaluate(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    #acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls,hist, mean_iu, iu, fwavacc

def evaluate_fast(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    
    return hist

def xrange(x):

    return iter(range(x))


def kl(x, y):
                X = tf.distributions.Categorical(probs=x)
                Y = tf.distributions.Categorical(probs=y)
                return tf.distributions.kl_divergence(X, Y)

def main():
    print("_______________________________________________")
    print("_______________________________________________")
    video_1, video_3, video_4, video_5,video_7,video_8,video_9,video_10,video_11, video_12,video_14, video_13=[],[],[],[],[],[],[],[],[],[],[],[]
    for filename in glob.glob('./13_zipped/*.jpg'):
        if(("13_6") in filename):
            video_13.append(filename)
    for filename in glob.glob('./Dataset/all_frames_patch_resize/*.jpg'): #assuming gif
        if(("1_1") in filename):
                    video_1.append(filename)
        if(("3_4") in filename):
                    video_3.append(filename)
        if(("4_2") in filename):
                    video_4.append(filename)
        if(("5_3") in filename):
                    video_5.append(filename)
        if(("7_6") in filename):
                    video_7.append(filename)
        if(("8_7") in filename):
                    video_8.append(filename)
        if(("9_2") in filename):
                    video_9.append(filename)
        if(("10_2") in filename):
                    video_10.append(filename)
        if(("11_3") in filename):
                    video_11.append(filename)
        if(("12_5") in filename):
                    video_12.append(filename)
        if(("14_7") in filename):
                    video_14.append(filename)


    
    with open('../frame_list_new.json') as f:
        merged_json = json.load(f)
    train_finetuning = video_1 +video_3+ video_4 +video_5+video_7+video_8+video_9+video_10+ video_11+video_12 +video_14
    validation= video_13
    cont=1

    valid_test=[]
    image_list_validation = []
    label_validation = []
    gaze_x_validation, gaze_y_validation = [],[]
    for filename in validation:
        valid_test.append(cont)
        cont+=1
    random.seed(10)
    random.shuffle(valid_test) #shuffle method
    random.shuffle(train_finetuning) #shuffle method
    
    for filename in valid_test[0:1]: 
        
        id_img ='13_6_tour_'+str(filename)
        x = merged_json[id_img]['gaze_x']
        y = merged_json[id_img]['gaze_y']
        current_label = merged_json[id_img]['looking_at']    
        
        #if(int(current_label)>5 and int(current_label)<15 ):
        image_list_validation.append('./13_zipped/13_6_tour_'+str(filename)+'.jpg')
        label_validation.append(int(current_label))
        gaze_x_validation.append(x)
        gaze_y_validation.append(y)
    

    image_list_train_finetuning = []
    label_train_finetuning = []
    gaze_x_train_finetuning, gaze_y_train_finetuning = [],[]

    
    all_sliding=[]
    for filename in glob.glob('./Dataset/Segmentation/resnet_segmentation_sliding_window_prob_all/1_1/*')[::10]: #assuming gif
        all_sliding.append(filename)
    
    for filename in glob.glob('./Dataset/Segmentation/resnet_segmentation_sliding_window_prob_all/4_2/*')[::10]: #assuming gif
        all_sliding.append(filename)
    
    for filename in glob.glob('./Dataset/Segmentation/resnet_segmentation_sliding_window_prob_all/12_5/*')[::10]: #assuming gif
        all_sliding.append(filename)
    
    for filename in glob.glob('./Dataset/Segmentation/resnet_segmentation_sliding_window_prob_all/3_4/*'):
        all_sliding.append(filename)
    for filename in glob.glob('./Dataset/Segmentation/resnet_segmentation_sliding_window_prob_all/5_3/*'): #assuming gif
        all_sliding.append(filename)
    for filename in glob.glob('./Dataset/Segmentation/resnet_segmentation_sliding_window_prob_all/7_6/*'): #assuming gif
        all_sliding.append(filename)
    for filename in glob.glob('./Dataset/Segmentation/resnet_segmentation_sliding_window_prob_all/8_7/*'): #assuming gif
        all_sliding.append(filename)
    for filename in glob.glob('./Dataset/Segmentation/resnet_segmentation_sliding_window_prob_all/9_2/*'): #assuming gif
        all_sliding.append(filename)
    for filename in glob.glob('./Dataset/Segmentation/resnet_segmentation_sliding_window_prob_all/10_2/*'): #assuming gif
        all_sliding.append(filename)
    for filename in glob.glob('./Dataset/Segmentation/resnet_segmentation_sliding_window_prob_all/11_3/*'): #assuming gif
        all_sliding.append(filename)
    for filename in glob.glob('./Dataset/Segmentation/resnet_segmentation_sliding_window_prob_all/14_7/*'): #assuming gif
        all_sliding.append(filename)
    print("all_sliding",len(all_sliding))
    
    looking_all_sliding =[]
    for element in all_sliding:
        
        name = element.split('/')[-1].split('.')[0].split('_sliding')[0]
       
        
        if(merged_json[name]['looking_at']!= 15):
            looking_all_sliding.append(element)
    print("looking_all_sliding",len(looking_all_sliding))
    all_sliding = looking_all_sliding
    random.shuffle(all_sliding)
    for filename in all_sliding: 
        
        id_img = filename.split('/')[-1].split('.')[0].replace('_sliding','')
        x = merged_json[id_img]['gaze_x']
        y = merged_json[id_img]['gaze_y']
        current_label = merged_json[id_img]['looking_at']    
        
        #if(int(current_label)>5 and int(current_label)<15 ):
        image_list_train_finetuning.append('./Dataset/all_frames/'+id_img+'.jpg')
        label_train_finetuning.append(int(current_label))
        gaze_x_train_finetuning.append(x)
        gaze_y_train_finetuning.append(y)
        

    batch_size = 1
    learning_rate = 0.001 #1e-3  #1e-3 
    train_finetuning_dataset = GazeDataset_resized(image_list_train_finetuning,label_train_finetuning,gaze_x_train_finetuning,gaze_y_train_finetuning,transforms=transforms_normal)
    train_finetuning_dataloader = DataLoader(train_finetuning_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = models.resnet18(pretrained=True)
    net = net.cuda() if device else net
    checkpoint = ""
    if(os.path.exists("./Dataset/resnet18-(2-13-6-valid)-checkpoint_best_resize300_epoch2.pt")): checkpoint = torch.load('./Dataset/resnet18-(2-13-6-valid)-checkpoint_best_resize300_epoch2.pt', map_location=device)
    num_ftrs = net.fc.in_features

    net.fc = nn.Linear(num_ftrs,16)
    net.fc = net.fc.cuda() if 1 else net.fc
    prev_epochs=0

    if(checkpoint):
        print("Checkpoint found")
        prev_epochs = checkpoint['epoch']
        prev_validation_acc = checkpoint['validation_acc']
        print("Prev_epochs: ",prev_epochs)
        net.load_state_dict(checkpoint['model_state_dict'])
        print("validation_acc: ",checkpoint['validation_acc'])


    
    class MyLinearLayer(nn.Module):
        """ Custom Linear layer but mimics a standard linear layer """
        def __init__(self, size_in, size_out):
            super().__init__()
            self.size_in, self.size_out = size_in, size_out
            weights = torch.Tensor(size_out, size_in)
            self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
            bias = torch.Tensor(size_out)
            self.bias = nn.Parameter(bias)
            self.linear_fc = net.fc

        def forward(self, x):

            permuted = x.permute(0,2,3,1)
            stacked=permuted.reshape(-1, 512)
            
            stacked_output = self.linear_fc(stacked) 
            return(stacked_output.view(*permuted.shape[:-1],16).permute(0,3,1,2))

    net_copy = torch.nn.Sequential(*(list(net.children())[:-2]))
    net_copy = nn.Sequential(
        net_copy,
        MyLinearLayer(512,16)
        ).to(device)  

    optimizer = optim.SGD(net_copy.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-10)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

    total_step = len(train_finetuning_dataset)
    print("total_step", total_step)
    n_epochs = 250
    writer = SummaryWriter()
    torch.autograd.set_detect_anomaly(True)
    
    total_iteration=0
    for epoch in range(1, n_epochs):     
        net_copy.train()
        batch_idx=0
        print(f'Epoch {epoch}\n')
       
        for batch_idx, (data, target, path) in enumerate(train_finetuning_dataloader):
            with torch.cuda.amp.autocast():

                total_iteration+=1 
                data=      data.to(device) 
                corrisponded_sliding = path[0].split('/')[-1].replace('.jpg','_sliding.npy.gz')
                optimizer.zero_grad()
                output = net_copy(data)
                sliding_pre_path = corrisponded_sliding.split('_tour')[0]
                f = gzip.GzipFile("./Dataset/Segmentation/resnet_segmentation_sliding_window_prob_all/"+sliding_pre_path+"/"+corrisponded_sliding , "r")
                loaded  = np.load(f,allow_pickle=True)
                loaded = F.interpolate(torch.from_numpy(loaded),(output.shape[2],output.shape[3]))               
                loss = F.kl_div(F.log_softmax(output[0]), loaded[0].to(device), reduction="mean")  #https://sidml.github.io/
                writer.add_scalar('Loss/train', torch.Tensor([loss]).unsqueeze(0).cpu(), total_iteration)
                loss.backward()          
                optimizer.step()
                if (batch_idx) % 9 == 0:   
                    print ('Epoch [{}/{}], Img. size [{}-{}] Step [{}/{}], Loss: {:.10f}' 
                            .format(epoch, n_epochs,output.shape[2],output.shape[3], batch_idx, total_step, loss.item()))
                    now = datetime.datetime.now()
                  
            
        
   
       
        
        print('INFERENCE')
        GLOBAL_ACC =0

        torch.save({
                    'model_state_dict': net_copy.state_dict(),
                    #'criterion_state_dict': criterion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch+1,
                    'lr': learning_rate,
                    'patch_size': 300,
                    'batch_size':batch_size,
                    
                    
                }, './finetuning_output_new_v3(1000_frame)/resnet18-FINETUNED_NEW_MODEL_epoch'+str(epoch)+'.pt')

            
            
    

if __name__ == "__main__":
    main()