This Project was made by Jie Zhang and Sujit Dityam for our seinor project 1 in semester 2 of 2024 and advised by Asst. Prof. Dr. Anilkumar Kothalil Gopalakrishnan.
THe full files are hosted on googel drive here:https://drive.google.com/drive/folders/1UfxYc7KgChK3LIHm3GbieADEVAJTC91h

[06 AI Enhanced Medical Lung Examination.pdf](https://github.com/user-attachments/files/21918667/06.AI.Enhanced.Medical.Lung.Examination.pdf)

ABSTRACT
The Pulmonary Nodule Detection is an application developed in python for doctors
and radiologists to inspect medical CT scans for diagnosing lung nodules which could lead to
cancer or other complications using a trained YOLO11s model from the Ultralytics python
library trained off labeled data from the open completion LUNA16. The YOLO model breaks
the CT scan into multiple sections, determines the existence of pulmonary nodules within the
sections above a certain confidence score and labels the sections accordingly. The application
also includes an AI chatbot which Users can train to help them find information about
pulmonary nodules using previously stored data fed into a GPT-2 Language model. All of
which is displayed to a user via a GUI made with python’s tkniter library.

METHODOLOGY
To Solve the difficulties with current lung diagnoses our Project will be both a
way to display and detect CT scans of patient's lungs while incorporating an AI
assistant. For detection of pulmonary nodules, we train an ai with the Yolo11s model
from Ultralytics’ python library with data from the open LUNA16 challenge where it’s
organizers publicly distributed CT scans of patients which were already categorized by
4 radiologists. The application allows for incremental training of the model to improve
accuracy so we decided on yolo as it was a fast model to train in such scenarios. We
tested 3 different yolo models (yolov5n, yolo v8n, yolo11s) and concluded yolo11s
was the most accurate model by having the best test scores. Yolo11s first developed by
Redmon, J. et. Al. And then improved to the 11th version by Ultralytics achieves
greater accuracy with fewer parameters through advancements in model design and
optimization techniques. The improved architecture allows for efficient feature
extraction and processing, resulting in higher mean Average Precision (mAP) on
datasets like COCO while using fewer parameters than other yolo models [7]. Unlike
older systems using classifiers or localizers to make detections using high scoring
regions, (Yolo models) apply a single neural network to the full image. This network
divides the image into regions and predicts bounding boxes and probabilities for each
region. These bounding boxes are weighted by the predicted probabilities [9]. Our
project trains Yolo11s on the CT scans from LUNA16 to look for signs of pulmonary
nodules in the body and display this annotation in the GUI. The GUI is made with
Python’s Tkinter library [10] which displays the CT scan and annotations in a window
next so slides and buttons to adjust parameters and access other functions

SYSTEM DESIGN
The system architecture for the Pulmonary Nodule Detection project is composed of
multiple integrated components. GUI made in Tkinter which displays all functions as well as
the label results from the Yolo model. The model is trained with 900 out of 1000 using
Pytorch which will take the necessary steps to created a Yolo model and calculate the
evaluations of the models then saving both the best and the latest trained models.
Additionally, CUDA is used so that a RTX 4090ti graphics card made by Nvidia may be used
for parallel processing of the training. Finally, this project uses local machine memory to
store model test scores, CT scans, and Chatbot data.
Lung Nodule Detection initiates data preprocessing by loading and normalizing CT
scan images, resizing and augmenting the data, and then training the YOLO model. It uses
annotated images for nodule detection with hyperparameter tuning for increased accuracy.
Following training, the model performs inference on new scans, detecting nodules as well as
confidence scores and bounding boxes. The results are displayed on the GUI for the users to
inspect. If any users find issue with the results the image can be put back into the training data
and used for incremental training to improve the model.
n the meantime, the AI Chatbot flow starts when the user inputs a query and is
preprocessed to normalize and tokenize the text. The preprocessed query is fed to the GPT-2
model, and it returns an answer based on data previously set or additionally inputted into a
text file. The answer is fed back to the user via the chatbot interface. Feedback on the
response, along with accuracy scores, is collected and stored to train the model for continuous
improvement. A feedback mechanism that optimizes the performance of the nodule detection
system and AI chatbot.
