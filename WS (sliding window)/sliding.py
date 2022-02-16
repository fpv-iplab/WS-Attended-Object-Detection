
import torch
from torch.autograd import Variable
import tensorflow as tf
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
import time
import copy
import os
import glob
import cv2
from sklearn import preprocessing
from PIL import Image, ImageOps
import datetime
import time
import numpy
import gzip
from torch.utils.tensorboard import SummaryWriter

from scipy.sparse import csc_matrix
import tensorflow as tf
from scipy import sparse
METHOD="resize"
SIZE = 300

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

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)
def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

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
       
        if(y-half_size<0):
            y=y+(half_size-y)
           
        if(x-half_size<0):
            x=x+(half_size-x)
            #print("x = 0 ")
        if(y+half_size>img_clear.shape[0]):
            #print("y > size ")
            y=img_clear.shape[0]-half_size
        if(x+half_size>img_clear.shape[1]):
            #print("x > size ")
            x=img_clear.shape[1]-half_size
        min_y=y-half_size
        min_x=x-half_size
        max_y=y+half_size
        max_x=x+half_size
        crop_img_small=    img_clear[min_y:max_y, min_x:max_x]
        #print(max_x,min_x, max_y,min_y, crop_img_small.shape[1],crop_img_small.shape[0])
    else:
        min_y=y-half_size
        min_x=x-half_size
        max_y=y+half_size
        max_x=x+half_size
        position=""
        if(y-half_size<0):
            min_y = 0
            #max_y = y+(min_y+y)
           
            position= 'up'
        if(x-half_size<0):
            min_x=0
            #max_x = x+(min_x+x)
          
            position= 'left'
        if(y+half_size>img_clear.shape[0]):
            max_y=img_clear.shape[0]
            #min_y = y-(max_y-y)
            
            position= 'down'
        if(x+half_size>img_clear.shape[1]):
            max_x=img_clear.shape[1]
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

transforms_normal = transforms.Compose(
[
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])


transforms_augmentation = transforms.Compose(
[
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    #La rotation viene fatta dopo la patch
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

class GazeDataset(Dataset):
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
        img = get_gaze_patch(img, self.gaze_x[i], self.gaze_y[i],SIZE,METHOD)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       
        if self.transforms is not None:
            image_from_array = Image.fromarray(img)
            image = self.transforms(image_from_array).numpy()
            
        return image,self.y[i]

class GazeDataset_augmentation(Dataset):
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.gaze_x[i] = self.gaze_x[i] + random.randint(-20, 20)
        self.gaze_y[i] = self.gaze_y[i] + random.randint(-20, 20)
        if self.transforms is not None:
            pil_image = Image.fromarray(img)
            pil_image = torchvision.transforms.RandomRotation((-10,10))(pil_image)
            open_cv_image = np.array(pil_image) 
            image_from_array = get_gaze_patch(open_cv_image, self.gaze_x[i], self.gaze_y[i],SIZE,METHOD)
            pil_image = Image.fromarray(image_from_array)
            image = self.transforms(pil_image).numpy()
            #image = np.rollaxis(image,0,3)
        return image,self.y[i]

def accuracy(out, labels):
        _,pred = torch.max(out, dim=1)
        return torch.sum(pred==labels).item()


def main():
    print("_______________________________________________")
    print("_______________________________________________")
   

    with open('./frame_list_new.json') as f:
        merged_json = json.load(f)
   
        
    print(len(merged_json))

    image_list_validation = []
    label_validation= []
    gaze_x_validation = []
    gaze_y_validation = []
    video_9 = []
    for filename in glob.glob('./all_frames/*.jpg'): 
        if(("9_2") in filename):
            video_9.append(filename)

    print("video_9", len(video_9)) 
    
    batch_size = 512

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = models.resnet18(pretrained=True)
    net = net.cuda() if device else net
    checkpoint = ""
    if(os.path.exists("./Dataset/resnet18-(2-13-6-valid)-checkpoint_best_resize300_epoch2.pt")): checkpoint = torch.load('./Dataset/resnet18-(2-13-6-valid)-checkpoint_best_resize300_epoch2.pt', map_location=device)

    criterion = nn.CrossEntropyLoss()

    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 16)
    net.fc = net.fc.cuda() if 1 else net.fc

    prev_epochs=0
    if(checkpoint):
        print("Checkpoint found")
        print(checkpoint.keys())
        prev_epochs = checkpoint['epoch']
        prev_validation_acc = checkpoint['validation_acc']
        print("Prev_epochs: ",prev_epochs,"valid_loss_min: ",checkpoint['valid_loss_min'])
        net.load_state_dict(checkpoint['model_state_dict'])
        criterion.load_state_dict(checkpoint['criterion_state_dict'])

        print("validation_acc: ",checkpoint['validation_acc'])
        print("train_accuracy: ",checkpoint['train_accuracy'])



    print("________________________________________")
    selected_for_sliding = "9_2"
    for i in range(1,len(video_9),100):
        start = time.time()
        print(i)
        image_list_validation=[]
        gaze_x_validation = []
        gaze_y_validation = []
        label_validation = []
        counter_x=15#int(SIZE/3)
        counter_y=15#int(SIZE/3)
        step = 32
        filename= "./all_frames/"+selected_for_sliding+"_tour_"+str(i)+".jpg"
        while(counter_x<2272 and counter_y<1278):

            image_list_validation.append(filename)
            label_validation.append(int(10))
            gaze_x_validation.append(counter_x)
            gaze_y_validation.append(counter_y)
            counter_x = counter_x +step
            if(counter_x>2272):
                counter_x=15
                counter_y = counter_y+step
        
        test_dataset =      GazeDataset(image_list_validation,label_validation,gaze_x_validation,gaze_y_validation,transforms=transforms_normal)
        test_dataloader =   DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        images_so_far = 1
        test = torch.FloatTensor(71, 40,16) 
        y_slide_count=0
        x_slide_count=0
        with torch.no_grad():
            net.eval()
            for data_t, target_t in (test_dataloader):
                data_t = data_t.to(device)
                
                target_t = target_t.to(device)
                outputs_t = net(data_t)
                pred_t = outputs_t
               
                for j in range(target_t.size()[0]):
                    
                    
                    if(images_so_far%100 ==0):
                        print(str(i)+")", str(images_so_far)+"/"+str(len(test_dataloader)*batch_size))
                    if(x_slide_count%71==0 and x_slide_count!= 0):
                        x_slide_count=0
                        y_slide_count+=1   
                    
                    test[x_slide_count][y_slide_count] =torch.nn.functional.softmax(pred_t[j])
                    #print(test[x_slide_count][y_slide_count])
                    x_slide_count+=1
                    images_so_far += 1
              

        cont=0   
        T  = (test)
        print("1",T.shape)
        T =T.permute(2,1,0)
        T = T.unsqueeze(0)
        T = T.unsqueeze(0)
        print("2",T.shape)
        #T = torch.nn.functional.interpolate(T,size=(16,1278,2272))
        T = T.squeeze(0)
        print("3",T.shape)
        f = gzip.GzipFile('./Dataset/Segmentation/resnet_segmentation_sliding_window_prob_all/'+selected_for_sliding+'/'+selected_for_sliding+'_tour_'+str(i)+'_sliding.npy.gz', "w")
        numpy.save(file=f, arr=T.cpu().numpy())
        f.close()
        end = time.time()
        print("Elapsed time: ", (end - start)/60, "batch:", batch_size)
           
    '''
    f = gzip.GzipFile('./Dataset/Segmentation/resnet_segmentation_sliding_window_prob_all/9_2/9_2_tour_1000_sliding.npy.gz', "r")
    loaded  = np.load(f,allow_pickle=True)
    print(loaded.shape)
    pred_t= loaded[0].argmax(0)
    mask = loaded[0].max(0) >0.5
    small_val=np.full((40,71),15)
    pred_t = torch.where(torch.from_numpy(mask),torch.from_numpy(pred_t),torch.from_numpy(small_val))
    save_image_for_test(pred_t,"./Dataset/Segmentation/resnet_segmentation_sliding_window_prob_all/")
    '''
if __name__ == "__main__":
    main()