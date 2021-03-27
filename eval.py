from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = './'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 48
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


checkpoint = "google_drive/MyDrive/checkpointsIeri/checkpoint_ssd300.pth.tar"
data_folder = 'google_drive/MyDrive/ColabNotebooks/Project/_provaSSD/test'



# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
import transforms, active_vision_dataset
#Include all instances
pick_trans = transforms.PickInstances(range(34))
TEST_PATH = "./google_drive/MyDrive/ColabNotebooks/Project/trainDataset"

#Include all instances
pick_trans = transforms.PickInstances(range(34))
test_dataset = active_vision_dataset.AVD(root=TEST_PATH,
                                    target_transform=pick_trans,
                                    scene_list=['Home_001_1'],
                                      fraction_of_no_box=-1)
  
test_loader = torch.utils.data.DataLoader(test_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=active_vision_dataset.collate
                          )


def evaluate(test_loader, model):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py
    
    from PIL import Image  
    import numpy as np    
    with torch.no_grad():
        # Batches
        for i, (images, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):
          #Code to avoid the "list index out of range" error (happens if the number
          #of last images is not perfeclty equal to batch_size
          numImagesLastBatch = -100
          #print("len:", len(test_loader))
          
          if i == len(test_loader)-1:
              numImagesLastBatch = len(test_dataset) - batch_size*(i)  
              print("Warning ", numImagesLastBatch)

          boxesList = list()
          labelList = list()
          diffList = list() 
          for j in range(batch_size):
            im = Image.fromarray(images[j])  

            box_id_diff = [b for b in labels[j][0]]      
            box = [l[0:4] for l in box_id_diff]
            
            #Boundary coordinates as requested
            for k in range(len(box)):  
              box[k][0] = box[k][0]/1920.0
              box[k][2] = box[k][2]/1920.0          
              box[k][1] = box[k][1]/1080.0
              box[k][3] = box[k][3]/1080.0 
          
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
          
            image, boxes, t_labels, difficulties = transform(im, box_tensor, label_tensor, diff_tensor, split='TEST')

            if j == 0:    
              image = image.unsqueeze(0)          
              imageBatch = image
            else:
              image = image.unsqueeze(0)
              imageBatch = torch.cat((imageBatch, image), 0)
                          
            boxesList.append(boxes.to(device))
            labelList.append(t_labels.to(device))
            diffList.append(difficulties.to(device))

            if j == numImagesLastBatch-1:
              j = batch_size
              break
          # ------------------------------            
          # Try to use the transformed stuff
          # ------------------------------
          images = imageBatch.to(device)
          boxes = boxesList
          labels = labelList
          difficulties = diffList

          # Forward prop.
          predicted_locs, predicted_scores = model(images)

          # Detect objects in SSD output
          det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                      min_score=0.01, max_overlap=0.45,
                                                                                      top_k=200)
          # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

          # Store this batch's results for mAP calculation
          #boxes = [b.to(device) for b in boxes]
          #labels = [l.to(device) for l in labels]
          #difficulties = [d.to(device) for d in difficulties]

          det_boxes.extend(det_boxes_batch)
          det_labels.extend(det_labels_batch)
          det_scores.extend(det_scores_batch)
          true_boxes.extend(boxes)
          true_labels.extend(labels)
          true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.8f' % mAP)



if __name__ == '__main__':
    evaluate(test_loader, model)