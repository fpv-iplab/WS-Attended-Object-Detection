
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
from torch.utils.tensorboard import SummaryWriter


METHOD="resize"
SIZE = 300

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

    merged_json = <path_to_annotation_file>



    batch_size = 8
    learning_rate = 1e-3 




    image_list_train = []
    label_train= []
    gaze_x_train = []
    gaze_y_train = []
    image_list_validation = []
    label_validation= []
    gaze_x_validation = []
    gaze_y_validation = []
    image_list_test = []
    label_test= []
    gaze_x_test = []
    gaze_y_test = []
    cont_file =0
    all_video={}
    train,validation,test = [],[],[]
    video_1,video_2,video_3=[],[],[]
    
    print("Video1", len(video_1), " video2", len(video_2), " video2", len(video_3))
    print("Train = video1 + video3")
    print("Test =[]")
    print("Validation = video2")
    random.shuffle(video_1)
    random.shuffle(video_2)
    random.shuffle(video_3)
    train= video_1 + video_3
    
    validation = video_2
    test =[]
    print("train", len(train), " validation", len(validation), " test: ",len(test))

    for filename in train: 
        id_img = filename.split('/')[3].replace('.jpg','')
        x = merged_json[id_img]['gaze_x']
        y = merged_json[id_img]['gaze_y']
        current_label = merged_json[id_img]['looking_at']
        if((x>0 and x<2272) and(y>0 and y<1278)):
            image_list_train.append(filename)
            label_train.append(int(current_label))
            gaze_x_train.append(x)
            gaze_y_train.append(y)
   
    for filename in validation: 
        id_img = filename.split('/')[3].replace('.jpg','')
        x = merged_json[id_img]['gaze_x']
        y = merged_json[id_img]['gaze_y']
        current_label = merged_json[id_img]['looking_at']
        if((x>0 and x<2272) and(y>0 and y<1278)):
            image_list_validation.append(filename)
            label_validation.append(int(current_label))
            gaze_x_validation.append(x)
            gaze_y_validation.append(y)
    
    for filename in test: 
        id_img = filename.split('/')[3].replace('.jpg','')
        x = merged_json[id_img]['gaze_x']
        y = merged_json[id_img]['gaze_y']
        current_label = merged_json[id_img]['looking_at']
        if((x>0 and x<2272) and(y>0 and y<1278)):
            image_list_test.append(filename)
            label_test.append(int(current_label))
            gaze_x_test.append(x)
            gaze_y_test.append(y)
    
    print("train", len(image_list_train), " validation", len(image_list_validation), " test: ",len(image_list_test))
    total = len(image_list_train) +len(image_list_validation) +len(image_list_test)
    
    print(str(int(len(image_list_train)*100/total))+"%",str(int(len(image_list_test)*100/total))+"%",str(int(len(image_list_validation)*100/total))+"%" )
    
    train_dataset = GazeDataset(image_list_train,label_train,gaze_x_train,gaze_y_train,transforms=transforms_normal)
    train_dataset_augmentation = GazeDataset_augmentation(image_list_train,label_train,gaze_x_train,gaze_y_train,transforms=transforms_augmentation)
    train_dataset =train_dataset+ train_dataset_augmentation
    test_dataset = GazeDataset(image_list_validation,label_validation,gaze_x_validation,gaze_y_validation,transforms=transforms_normal)
   
    print("train dataset size: {}".format(len(train_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    images, labels = next(iter(train_dataloader)) 
    out = torchvision.utils.make_grid(images)


  
    display_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    images_to_display, labels_to_display = next(iter(display_data_loader)) 
    for index, element in enumerate(images_to_display):
        open_cv_image = np.array(element)
        image = np.rollaxis(open_cv_image,0,3)
        plt.imshow(image)
        plt.title(labels_to_display[index])
        plt.show()
    
    net = models.resnet18(pretrained=True)
    net = net.cuda() if device else net
    checkpoint = ""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 16)
    net.fc = net.fc.cuda() if 1 else net.fc
    prev_epochs=0
    if(checkpoint):
        print("Checkpoint found")
        #print(checkpoint.keys())
        prev_epochs = checkpoint['epoch']
        print("Prev_epochs: ",prev_epochs,"valid_loss_min: ",checkpoint['valid_loss_min'])
        net.load_state_dict(checkpoint['model_state_dict'])
        criterion.load_state_dict(checkpoint['criterion_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("validation_acc: ",checkpoint['validation_acc'])
        print("train_accuracy: ",checkpoint['train_accuracy'])
                


   
    
    epochs_done=prev_epochs 
    epochs_done = epochs_done+1
    n_epochs =  10 + epochs_done 
    print_every = 20
  
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(train_dataloader)
    f = open('log_resnet.txt', 'a')
    writer = SummaryWriter("my_experiment")

    for epoch in range(epochs_done, n_epochs):
        
        running_loss = 0.0
        correct = 0
        total=0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(train_dataloader):
            data_=      data_.to(device) 
            target_ =   target_.to(device)
            optimizer.zero_grad()
            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==target_).item()
            total += target_.size(0)
            if (batch_idx) % print_every == 0:
                
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}, Method: {}, Size: {}' 
                    .format(epoch, n_epochs, batch_idx, total_step, loss.item(), (100 * correct/total), METHOD, SIZE))
                now = datetime.datetime.now()
                f.write(now.strftime("%Y-%m-%d %H:%M:%S")+ ' Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}' 
                    .format(epoch, n_epochs, batch_idx, total_step, loss.item(), (100 * correct/total))+"\n")
     
                writer.add_scalar("Loss", loss.item(), batch_idx)
                writer.add_scalar("Correct", correct, batch_idx)
                writer.add_scalar("Accuracy", (100 * correct/total), batch_idx)

               
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss/total_step)
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
        train_accuracy =(100 * correct/total)
        
        batch_loss = 0
        total_t=0
        correct_t=0
        
        with torch.no_grad():
            net.eval()
            for data_t, target_t in (test_dataloader):
                data_t = data_t.to(device)
                target_t = target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _,pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t==target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t/total_t)
            val_loss.append(batch_loss/len(test_dataloader))
            print("batch_loss: ",batch_loss," valid_loss_min: ", valid_loss_min)
            network_learned = batch_loss < valid_loss_min
            validation_acc=(100 * correct_t/total_t)
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')
            
            if network_learned:
                valid_loss_min = batch_loss
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'lr': learning_rate,
                    'patch_size': SIZE,
                    'batch_size':batch_size,
                    'valid_loss_min':valid_loss_min,
                    'validation_acc':validation_acc,
                    'train_accuracy':train_accuracy,
                }, 'resnet18-checkpoint_best_'+str(METHOD)+str(SIZE)+'.pt')
                print('Improvement-Detected, saving best model')        
            torch.save({
                    'model_state_dict': net.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'lr': learning_rate,
                    'patch_size': SIZE,
                    'batch_size':batch_size,
                    'valid_loss_min':valid_loss_min,
                    'validation_acc':validation_acc,
                    'train_accuracy':train_accuracy,
                }, 'resnet18-checkpoint_last_'+str(METHOD)+str(SIZE)+'.pt')
            print('Saving last model')       
            
        net.train()

if __name__ == "__main__":
    main()