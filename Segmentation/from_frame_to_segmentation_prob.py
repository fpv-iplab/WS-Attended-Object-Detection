
import datetime
from scipy import sparse
import numpy as np
import glob
import cv2
import json
import gzip
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import * 
from torch.utils.data import Dataset, DataLoader
from skimage.filters.rank import modal
from skimage.morphology import rectangle
from scipy.ndimage import label, generate_binary_structure
from collections import Counter
import PIL
import tensorflow as tf
from PIL import Image


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
        #return torch.add(stacked, self.bias)





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


transforms_normal = transforms.Compose(
[
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
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
  cv2.imwrite(pred_t+".png", dest) 

with open('./Dataset/frame_list_new.json') as f:
    merged_json = json.load(f)


image_list_validation = []
label_validation= []
gaze_x_validation = []
gaze_y_validation = []

cont_file =0
all_video={}
train,validation,test = [],[],[]
video_1,video_2,video_3,video_4,video_6, video_13=[],[],[],[],[],[]
for filename in glob.glob('./Dataset/all_frames/*'): #assuming gif
    #print(filename)

    if(("2_2") in filename):
        video_2.append(filename)
     
    if(("13_6") in filename):
        video_13.append(filename)
    if(("6_5") in filename):
        video_6.append(filename)

print( " video2", len(video_2), " video13", len(video_13), " video_6", len(video_6))

image_list_validation=[]
label_validation=[]
gaze_x_validation=[]
gaze_y_validation=[]

   
validation= video_13
cont=1

valid_test=[]
for filename in validation:
    valid_test.append(cont)
    cont+=1

for filename in valid_test[2999:3000]: 
    
    id_img ='13_6_tour_'+str(filename)
    try:
        x = merged_json[id_img]['gaze_x']
        y = merged_json[id_img]['gaze_y']
        current_label = merged_json[id_img]['looking_at']    
        
        #if((x>0 and x<2278) and (y>0 and y<1278)):
        image_list_validation.append('./Dataset/all_frames/13_6_tour_'+str(filename)+'.jpg')
        label_validation.append(int(current_label))
        gaze_x_validation.append(x)
        gaze_y_validation.append(y)
    except:
        continue
  
print(len(image_list_validation))
print((image_list_validation[0]))
batch_size = 1
learning_rate = 1e-3 



test_dataset = GazeDataset_resized(image_list_validation,label_validation,gaze_x_validation,gaze_y_validation,transforms=transforms_normal)

print("valid dataset size: {}".format(len(test_dataset)))

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = models.resnet18(pretrained=True)
net = net.cuda() if device else net
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs,16)


net_copy = torch.nn.Sequential(*(list(net.children())[:-2]))
net_copy = nn.Sequential(
    net_copy,
    MyLinearLayer(512,16)
    ).to(device)  
checkpoint = ""

if(os.path.exists("./Dataset/Segmentation/finetuning_output_new_v4_experiment/resnet18-FINETUNED_NEW_MODEL_epoch10.pt")): checkpoint = torch.load("./Dataset/Segmentation/finetuning_output_new_v4_experiment/resnet18-FINETUNED_NEW_MODEL_epoch10.pt", map_location=device)


if(checkpoint):
    print("Checkpoint found")
    prev_epochs = checkpoint['epoch']
    print("Prev_epochs: ",prev_epochs)
    net_copy.load_state_dict(checkpoint['model_state_dict'])


prev_epochs=0


n_epochs = 1



images_so_far = 0
features=None
segmented_image_form_resnet ={}
with torch.no_grad():
    net_copy.eval()
    start = time.time()
    for batch_idx, (data_t, target_t, path_t) in enumerate(test_dataloader):      
       
        
        data_t = data_t.to(device)
        target_t = target_t.to(device)
    
        output = net_copy(data_t)
      
        print("output", output.shape)
        for j in range(output.size()[0]):
          name= path_t[j].split('/')[-1].replace('.jpg','')
          print(path_t[j].split('/')[-1].replace('.jpg',''))
          end = time.time()
          print((end - start)/3600)
          f = gzip.GzipFile('<segmentation_prob>/post_finetuning_new_v4_experiment/13_6/'+name+'_antonino_method.gz', "w")
          np.save(file=f, arr=output.cpu().numpy())
          f.close()

          images_so_far+=1
        
    
