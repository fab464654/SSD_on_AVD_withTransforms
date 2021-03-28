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
pick_trans = transforms.PickInstances(range(33))
TEST_PATH = "./google_drive/MyDrive/ColabNotebooks/Project/trainDataset"

test_dataset = active_vision_dataset.AVD(root=TEST_PATH,
                                    target_transform=pick_trans,
                                    scene_list=['Home_003_1'],
                                      fraction_of_no_box=-1,
                                      split = "TEST")
  
test_loader = torch.utils.data.DataLoader(test_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=active_vision_dataset.collate_fn
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
     
    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
          images = images.to(device)  # (N, 3, 300, 300)
          
          """
          #CODE TO SEE/SAVE THE ACTUAL IMAGES
          from torchvision import transforms
          x = images[0]
          mean = [0.485, 0.456, 0.406]
          std = [0.229, 0.224, 0.225]  
          z = x * torch.tensor(std).to(device).view(3, 1, 1)
          z = z + torch.tensor(mean).to(device).view(3, 1, 1)

          trans = transforms.ToPILImage()
          immagine = (trans(z).convert("RGB"))
          immagine.save("image.jpg")
          """
          
          # Forward prop.
          predicted_locs, predicted_scores = model(images)
          print("  ", predicted_locs,"   ", predicted_scores)
          # Detect objects in SSD output
          det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                      min_score=0.01, max_overlap=0.45,
                                                                                      top_k=200)
          # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

          # Store this batch's results for mAP calculation
          boxes = [b.to(device) for b in boxes]
          labels = [l.to(device) for l in labels]
          difficulties = [d.to(device) for d in difficulties]

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
