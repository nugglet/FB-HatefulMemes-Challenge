# Facebook Hateful Memes Challenge

## Key Ideas from the Paper
The PDF version of the Facebook research paper can be found in the `References` folder.

This paper attempts to understand the subtle semantics in multi-modal communication through memes, which combines text and photographic pictures in a single image.

The challenge aims to use transfer learning from pre-trained multimodal models and finetune them to the task. This challenge follows the following definitions:
- Hate speech: A direct or indirect attack on people based on characteristics. Attacks are defined as mocking, violent or dehumanizing statements, or speech that calls for exclusion and discrimination. The exceptions to this rule are: Attacking individuals if the attack is not based on any of the protected characteristics listed in the definition or Attacking groups perpetrating hate (e.g. terrorist groups)

### Notes on the Dataset
- The data is meme image + pre-extracted text. 
- Each meme is labelled with a binary classification (hateful or non-hateful)
- Total 10k memes
- Benign Confounders (Contrasting Examples): For every hateful meme, an alternative non-hateful version of the meme is created (with either opposite captioning or alternative images)
- The dataset comprises five different types of memes: multimodal hate, where benign confounders were found for both modalities, unimodal hate where one or both modalities were already hateful on their own, benign image and benign text confounders and finally random not-hateful examples
- Data is split as such:
    - dev (5%), test (10%), train (75%)
    - dev and test: 40% mm hate, 10% uni hate, 20% benign text, 20% benign img, 10% random benign


### Models
The paper uses a variety of models of type: unimodal, multimodal (unimodally pretrained), multimodal (multimodal pretrained)

Image Encoders:
1. Resnet-152 layer res-5c (Image-grid)
2. FasterRCNN layer fc6, finetuned using fc7 weights (Image-region)

Text: BERT

Method of combination for multimodal + unimodal pretrained:
1. take mean of score output from unimodal image and text encoders (late fusion)
2. Concat features from img and text encoders, train MLP on top

Multimodal + Multimodal pretrained models:
- ViLBERT CC
- Visual BERT COCO
- supervised multimodal bitransformers (MMBT-grid and MMBT-region)

### Model Evaluation
#### Area under the Receiver Operatic Characteristic Curve (ROC AUC)
ROC is a probability curve and AUC represents the degree or measure of separability. It tells how much the model is capable of distinguishing between classes.

- Baseline Accuracy: 84.7% (Human Accuracy)



## Research Checkpoints
The following is an outline of the steps I plan to take to solve the problem of hateful meme classification.

1. [x] Initialize project (Dataset, environment etc.)
2. [x] Data Pre-processing
    - [x] Loading image data
    - [x] Loading text data
    - [x] Handling variable sized data
3. Try Unimodal Pre-trained Models
    - Resnet + BERT (which resnet to use? will it make a difference?) [from the paper]
        - [] Try Resnet + GPT2
    - [] FasterRCNN + BERT (What abt other RCNN like masked?) [from the paper]
    - [] Maybe try other segmentation models + BERT (YOLO? Bc its fast and understands generalized object representation)
4. Try Multimodal + Unimodal Pre-trained Models
    -
5. Try Multimodal + Multimodal Pre-trained Models
   - ViLBERT [From the paper]
   - Visual BERT COCO [From the paper]
   -
6. Finetuning
7. Evaluation

## Links and Resources
On the HatefulMemes Challenge:
- https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/
- https://hatefulmemeschallenge.com/#about
- https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes

Other Solutions:
- https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset/code
- 