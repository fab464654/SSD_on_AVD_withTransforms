import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
import numpy as np

# Data parameters
data_folder = 'google_drive/MyDrive/ColabNotebooks/Project/GT' # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = "google_drive/MyDrive/checkpointsIeri/checkpoint_ssd300.pth.tar"  # path to model checkpoint, None if none
batch_size = 9  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 5  # print training status every __ batches
lr = 5e-4  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    #import active_vision_dataset_processing.data_loading
    import transforms, active_vision_dataset

    #Include all instances
    pick_trans = transforms.PickInstances(range(34))

    TRAIN_PATH = "./google_drive/MyDrive/ColabNotebooks/Project/trainDataset"

    
    train_dataset = active_vision_dataset.AVD(root=TRAIN_PATH, train=True,
                                        target_transform=pick_trans,
                                        scene_list=['Home_001_1',                                                    
                                                    'Home_002_1',
                                                    'Home_003_1',                                                    
                                                    'Home_004_1',
                                                    'Home_005_1',
                                                    'Home_006_1',
                                                    'Home_007_1',
                                                    'Home_008_1',
                                                    'Home_014_1',
                                                    'Home_011_1',
                                                    'Home_010_1',
                                                    'Office_001_1'],
                                          fraction_of_no_box=-1)

    
    """         
    #I wasn't using the collate function and the transform method, so this is a new try:
    from PIL import Image
    
    for k in range(len(train_dataset)):      
      images_pre,labels_pre = train_dataset[k]
      
      # Read image
      image = Image.fromarray(np.uint8(images_pre)).convert('RGB')
      #image = Image.open(images_pre[k], mode='r')
      #image = image.convert('RGB')
      

      # Read objects in this image (bounding boxes, labels, difficulties)
      box_id_diff = [b for b in labels_pre[k][0]]     
      box = [l[0:4] for l in box_id_diff]
      print(box) 

      #Boundary coordinates as requested
      for k in range(len(box)):  
        box[k][0] = box[k][0]/1920.0
        box[k][2] = box[k][2]/1920.0          
        box[k][1] = box[k][1]/1080.0
        box[k][3] = box[k][3]/1080.0 
      
      boxes = torch.FloatTensor(box)  # (n_objects, 4)
      print(boxes)
      labels = torch.LongTensor(labels_pre[1])  # (n_objects)
      difficulties = torch.ByteTensor(labels_pre[2])  # (n_objects)
    
      # Apply transformations
      image, boxes, labels, difficulties = utils.transform(image, boxes, labels, difficulties, split='TRAIN')
      
      # Update the dataset's elements
      

    """
    #collate_fn=active_vision_dataset.collate
    train_loader = torch.utils.data.DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=active_vision_dataset.collate, num_workers=workers,
                              pin_memory=True)
    """
   

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
   """

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    # Epochs
    for epoch in range(start_epoch, epochs):
        
        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)
      


def train(train_loader, model, criterion, optimizer, epoch):
    
    
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    
    # Batches
    for i, (images, labels) in enumerate(train_loader):

        
        #Pre-processing:     
         
        from torchvision import transforms as transf
        preprocess = transf.Compose([
                      transf.ToPILImage(),
                      transf.Resize(300),
                      transf.CenterCrop(300),
                      transf.ToTensor(),  
                      transf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                  ])
        
        boxesList = list()
        labelList = list()
        diffList = list()
        
        for j in range(batch_size): 

          if j == 0:              
            input_tensor = preprocess(images[j])
            input_tensor = input_tensor.unsqueeze(0)
            input_batch = input_tensor
          else:
            input_tensor = preprocess(images[j])
            #print(input_tensor)
            input_tensor = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
            #print(input_tensor.shape)
            input_batch = torch.cat((input_batch, input_tensor), 0)
            #print("shape images: ",input_batch.shape) 
          
          
         
          # In the Active Vision Dataset we have this formatting:
          # [xmin ymin xmax ymax instance_id difficulty]
            
          """          From the Tutorial:           
          Since the number of objects in any given image can vary, we can't use a fixed 
          size tensor for storing the bounding boxes for the entire batch of N images.

          Therefore, ground truth bounding boxes fed to the model must be a list of 
          length N, where each element of the list is a Float tensor of dimensions
          N_o, 4, where N_o is the number of objects present in that particular image.

          Therefore, ground truth labels fed to the model must be a list of length N, 
          where each element of the list is a Long tensor of dimensions N_o, where N_o 
          is the number of objects present in that particular image.
          """
          #Prints to test
          #print(j)
          box_id_diff = [b for b in labels[j][0]]      
          box = [l[0:4] for l in box_id_diff]

          #print('before:',box) #To check

          #Boundary coordinates as requested
          for k in range(len(box)):  
            box[k][0] = box[k][0]/1920.0
            box[k][2] = box[k][2]/1920.0          
            box[k][1] = box[k][1]/1080.0
            box[k][3] = box[k][3]/1080.0 

          #print('after:',box) #To check

         
          box_tensor = torch.FloatTensor(box)

          
          if j == 0:            
            box_list = [box_tensor]
          else:
            box_list.append(box_tensor)               

          label = [l[4] for l in box_id_diff]
          label_tensor = torch.LongTensor(label)
          if j == 0: 
            label_list = [label_tensor]
          else:
            label_list.append(label_tensor)        

          diff = [l[5] for l in box_id_diff]
          for m in range(len(diff)):
            if diff[m] > 3:
              diff[m] = 1
            else:
              diff[m] = 0

          diff_tensor = torch.ByteTensor(diff)
          if j == 0: 
            diff_list = [diff_tensor]
          else:
            diff_list.append(diff_tensor)

          # ------------------------------            
          # Apply transformations TO TEST
          # ------------------------------
          """
          from PIL import Image
          print(images[j])
          img = Image.fromarray(images[j])
          """
          from PIL import Image
          from matplotlib import cm    
          import numpy as np     
          
          im = Image.fromarray(images[j])     
          
          image, boxes, t_labels, difficulties = transform(im, box_tensor, label_tensor, diff_tensor, split='TRAIN')

          if j == 0:    
            image = image.unsqueeze(0)          
            imageBatch = image
          else:
            image = image.unsqueeze(0)
            imageBatch = torch.cat((imageBatch, image), 0)

          boxesList.append(boxes.to(device))
          labelList.append(t_labels.to(device))
          diffList.append(difficulties.to(device))

        #print("list of boxes:",box_list)
        #print("list of labels:", label_list)
        """
        images = input_batch.to(device)  # (batch_size (N), 3, 300, 300)
        print(images.shape)
        boxes = box_list
        labels = label_list
        difficulties = diff_list
        """
        # ------------------------------            
        # Try to use the transformed stuff
        # ------------------------------
        images = imageBatch.to(device)
        boxes = boxesList

        # Forward prop.        
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        #Prints to check the dimensions
        #print(predicted_locs.shape)    #correct    
        #print(predicted_scores.shape)  #correct  

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labelList)  # scalar
        
        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()
        
        # Print status
        if i % print_freq == 0:          
            print('Epoch: [{0}][{1}/{2}]\t'                  
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), loss=losses))
            """
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
            """                                                        
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()
