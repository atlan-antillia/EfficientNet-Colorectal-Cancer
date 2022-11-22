<h2>EfficientNet-Colorectal-Cancer (Updated: 2022/11/23)</h2>
<a href="#1">1 EfficientNetV2 Colorectal Cancer Classification </a><br>
<a href="#1.1">1.1 Clone repository</a><br>
<a href="#1.2">1.2 Prepare Colorectal-Cancer dataset</a><br>
<a href="#1.3">1.3 Install Python packages</a><br>
<a href="#2">2 Python classes for Colorectal-Cancer Classification</a><br>
<a href="#3">3 Pretrained model</a><br>
<a href="#4">4 Train</a><br>
<a href="#4.1">4.1 Train script</a><br>
<a href="#4.2">4.2 Training result</a><br>
<a href="#5">5 Inference</a><br>
<a href="#5.1">5.1 Inference script</a><br>
<a href="#5.2">5.2 Sample test images</a><br>
<a href="#5.3">5.3 Inference result</a><br>
<a href="#6">6 Evaluation</a><br>
<a href="#6.1">6.1 Evaluation script</a><br>
<a href="#6.2">6.2 Evaluation result</a><br>


<h2>
<a id="1">1 EfficientNetV2 Colorectal Cancer Classification</a>
</h2>

 This is an experimental EfficientNetV2 Colorectal Cancer Classification project based on <b>efficientnetv2</b> in <a href="https://github.com/google/automl">Brain AutoML</a>.
<br>

The image dataset <b>CRC-VAL-HE-7K</b> used here has been taken from the following website:<br>
Zenodo: 100,000 histological images of human colorectal cancer and healthy tissue<br>
https://zenodo.org/record/1214456#.Y3sVDHZBy3A
<br>

<br>We use python 3.8 and tensorflow 2.8.0 environment on Windows 11.<br>
<br>

<h3>
<a id="1.1">1.1 Clone repository</a>
</h3>
 Please run the following command in your working directory:<br>
<pre>
git clone https://github.com/atlan-antillia/EfficientNet-Colorectal-Cancer.git
</pre>
You will have the following directory tree:<br>
<pre>
.
├─asset
└─projects
    └─Colorectal-Cancer
        ├─eval
        ├─evaluation
        ├─inference
        ├─models
        ├─CRC-VAL-HE-7K-Images
        └─test
</pre>
<h3>
<a id="1.2">1.2 Prepare Colorectal Cancer dataset</a>
</h3>
<b>1, Download dataset</b> <br>
The image dataset <b>CRC-VAL-HE-7K</b> used here has been taken from the following website:<br>
<a href="https://zenodo.org/record/1214456#.Y3sVDHZBy3A">Zenodo:100,000 histological images of human colorectal cancer and healthy tissue</a>
<br>
<pre>
CRC-VAL-HE-7K
├─ADI
├─BACK
├─DEB
├─LYM
├─MUC
├─MUS
├─NORM
├─STR
└─TUM
</pre>

<b>2, Convert jpg files</b> <br>
 For simplicity, we have removed <b>BACK</b> and <b>DEB</b> folder from original dataset. and converted the original tif files
 to jpg files by using <a href="./projects/Colorectal-Cancer/jpg_converter.py">jpg_converter.py</a> script.<br>
<pre>
CRC-VAL-HE-7K-jpg-master
├─ADI
├─LYM
├─MUC
├─MUS
├─NORM
├─STR
└─TUM
</pre>

<b>3, Split master dataset</b> <br>
 We have split the <b>CRC-VAL-HE-7K-jpg-master</b> to <b>train</b> and <b>test</b> by using 
 <a href="./projects/Colorectal-Cancer/split_master.py">split_master.py</a> script.<br>
Finally, we have created the <b>CRC-VAL-HE-7K-images</b> dataset from the <b>CRC-VAL-HE-7K-jpg-master</b> dataset.<br>
<pre>
CRC-VAL-HE-7K-images
├─test
│  ├─ADI
│  ├─LYM
│  ├─MUC
│  ├─MUS
│  ├─NORM
│  ├─STR
│  └─TUM
└─train
    ├─ADI
    ├─LYM
    ├─MUC
    ├─MUS
    ├─NORM
    ├─STR
    └─TUM
</pre>
 
The distribution of the number of images in the dataset is the following:<br>
<img src="./projects/Colorectal-Cancer/_CRC-VAL-HE-7K-Images_.png" width="740" height="auto"><br>
<br>

CRC-VAL-HE-7K-images/train/ADI:<br>
<img src="./asset/CRC-VAL-HE-7K-images_train_ADI.png" width="840" height="auto">
<br>
<br>
CRC-VAL-HE-7K-images/train/LYM:<br>
<img src="./asset/CRC-VAL-HE-7K-images_train_LYM.png" width="840" height="auto">
<br>
<br>
CRC-VAL-HE-7K-images/train/MUC:<br>
<img src="./asset/CRC-VAL-HE-7K-images_train_MUC.png" width="840" height="auto">
<br>
<br>
CRC-VAL-HE-7K-images/train/MUS:<br>
<img src="./asset/CRC-VAL-HE-7K-images_train_MUS.png" width="840" height="auto">
<br>

<br>
CRC-VAL-HE-7K-images/train/NORM:<br>
<img src="./asset/CRC-VAL-HE-7K-images_train_NORM.png" width="840" height="auto">
<br>
<br>
CRC-VAL-HE-7K-images/train/STR:<br>
<img src="./asset/CRC-VAL-HE-7K-images_train_STR.png" width="840" height="auto">
<br>
<br>
CRC-VAL-HE-7K-images/train/TUM:<br>
<img src="./asset/CRC-VAL-HE-7K-images_train_TUM.png" width="840" height="auto">
<br>

<h3>
<a id="#1.3">1.3 Install Python packages</a>
</h3>
Please run the following commnad to install Python packages for this project.<br>
<pre>
pip install -r requirements.txt
</pre>
<br>

<h2>
<a id="2">2 Python classes for Colorectal-Cancer Classification</a>
</h2>
We have defined the following python classes to implement our Colorectal-Cancer Classification.<br>
<li>
<a href="./ClassificationReportWriter.py">ClassificationReportWriter</a>
</li>
<li>
<a href="./ConfusionMatrix.py">ConfusionMatrix</a>
</li>
<li>
<a href="./CustomDataset.py">CustomDataset</a>
</li>
<li>
<a href="./EpochChangeCallback.py">EpochChangeCallback</a>
</li>
<li>
<a href="./EfficientNetV2Evaluator.py">EfficientNetV2Evaluator</a>
</li>
<li>
<a href="./EfficientNetV2Inferencer.py">EfficientNetV2Inferencer</a>
</li>
<li>
<a href="./EfficientNetV2ModelTrainer.py">EfficientNetV2ModelTrainer</a>
</li>
<li>
<a href="./FineTuningModel.py">FineTuningModel</a>
</li>

<li>
<a href="./TestDataset.py">TestDataset</a>
</li>

<h2>
<a id="3">3 Pretrained model</a>
</h2>
 We have used pretrained <b>efficientnetv2-b0</b> model to train Colorectal-Cancer Classification FineTuning Model.
Please download the pretrained checkpoint file from <a href="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-b0.tgz">efficientnetv2-b0.tgz</a>, expand it, and place the model under our top repository.

<pre>
.
├─asset
├─efficientnetv2-b0
└─projects
    └─Colorectal-Cancer
  ...
</pre>

<h2>
<a id="4">4 Train</a>

</h2>
<h3>
<a id="4.1">4.1 Train script</a>
</h3>
Please run the following bat file to train our Colorectal-Cancer Classification efficientnetv2 model by using
<a href="./projects/Colorectal-Cancer/Colorectal_Cancer_Images/train">Resampled_Kidney_Simpler_Disease_Images/train</a>.
<pre>
./1_train.bat
</pre>
<pre>
rem 1_train.bat
python ../../EfficientNetV2ModelTrainer.py ^
  --model_dir=./models ^
  --eval_dir=./eval ^
  --model_name=efficientnetv2-b0  ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../efficientnetv2-b0/model ^
  --optimizer=rmsprop ^
  --image_size=224 ^
  --eval_image_size=224 ^
  --data_dir=./CRC-VAL-HE-7K-Images/train ^
  --data_augmentation=True ^
  --valid_data_augmentation=True ^
  --fine_tuning=True ^
  --monitor=val_loss ^
  --learning_rate=0.0001 ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.2 ^
  --num_epochs=50 ^
  --batch_size=8 ^
  --patience=10 ^
  --debug=True  
</pre>
, where data_generator.config is the following:<br>
<pre>
; data_generation.config

[training]
validation_split   = 0.2
featurewise_center = Fale
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 90
horizontal_flip    = True
vertical_flip      = True
width_shift_range  = 0.1
height_shift_range = 0.1
shear_range        = 0.01
zoom_range         = [0.8, 1.2]
data_format        = "channels_last"

[validation]
validation_split   = 0.2
featurewise_center = False
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 90
horizontal_flip    = True
vertical_flip      = True
width_shift_range  = 0.1
height_shift_range = 0.1
shear_range        = 0.01
zoom_range         = [0.8, 1.2]
data_format        = "channels_last"
</pre>

<h3>
<a id="4.2">4.2 Training result</a>
</h3>

This will generate a <b>best_model.h5</b> in the models folder specified by --model_dir parameter.<br>
Furthermore, it will generate a <a href="./projects/Colorectal-Cancer/eval/train_accuracies.csv">train_accuracies</a>
and <a href="./projects/Colorectal-Cancer/eval/train_losses.csv">train_losses</a> files
<br>
Training console output:<br>
<img src="./asset/Colorectal_Cancer_train_at_epoch_50_1122.png" width="740" height="auto"><br>
<br>
Train_accuracies:<br>
<img src="./projects/Colorectal-Cancer/eval/train_accuracies.png" width="640" height="auto"><br>

<br>
Train_losses:<br>
<img src="./projects/Colorectal-Cancer/eval/train_losses.png" width="640" height="auto"><br>

<br>
<h2>
<a id="5">5 Inference</a>
</h2>
<h3>
<a id="5.1">5.1 Inference script</a>
</h3>
Please run the following bat file to infer the Colorectal Cancer in test images by the model generated by the above train command.<br>
<pre>
./2_inference.bat
</pre>
<pre>
rem 2_inference.bat: modified eval_image_size to be 224
python ../../EfficientNetV2Inferencer.py ^
  --model_name=efficientnetv2-b0  ^
  --model_dir=./models ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.2 ^
  --image_path=./test/*.jpeg ^
  --eval_image_size=224 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --infer_dir=./inference ^
  --debug=False 
</pre>
<br>
label_map.txt:
<pre>
ADI
LYM
MUC
MUS
NORM
STR
TUM
</pre>
<br>
<h3>
<a id="5.2">5.2 Sample test images</a>
</h3>

Sample test images generated by <a href="./projects/Colorectal-Cancer/create_test_dataset.py">create_test_dataset.py</a> 
from <a href="./projects/Colorectal-Cancer/Colorectal_Images/test">Colorectal_Imagess/test</a>.
<br>
<img src="./asset/CRC-VAL-HE-7K-images_test.png" width="840" height="auto"><br>

<h3>
<a id="5.3">5.3 Inference result</a>
</h3>
This inference command will generate <a href="./projects/Colorectal-Cancer/inference/inference.csv">inference result file</a>.
<br>
<br>
Inference console output:<br>
<img src="./asset/Colorectal_Cancer_infer_at_epoch_50_1122.png" width="740" height="auto"><br>
<br>

Inference result (inference.csv):<br>
<img src="./asset/Colorectanl_Cancer_inference_at_epoch_50_1122.png" width="740" height="auto"><br>
<br>
<h2>
<a id="6">6 Evaluation</a>
</h2>
<h3>
<a id="6.1">6.1 Evaluation script</a>
</h3>
Please run the following bat file to evaluate <a href="./projects/Colorectal-Cancer/CRC-VAL-HE-7K-Images/test">
CRC-VAL-HE-7K-Images/test</a> by the trained model.<br>
<pre>
./3_evaluate.bat
</pre>
<pre>
rem 3_evaluate.bat
python ../../EfficientNetV2Evaluator.py ^
  --model_name=efficientnetv2-b0  ^
  --model_dir=./models ^
  --data_dir=./CRC-VAL-HE-7K-Images/test ^
  --evaluation_dir=./evaluation ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.2 ^
  --eval_image_size=224 ^
  --mixed_precision=True ^
  --debug=False 
</pre>
<br>

<h3>
<a id="6.2">6.2 Evaluation result</a>
</h3>

This evaluation command will generate <a href="./projects/Colorectal-Cancer/evaluation/classification_report.csv">a classification report</a>
 and <a href="./projects/Colorectal-Cancer/evaluation/confusion_matrix.png">a confusion_matrix</a>.
<br>
<br>
Evaluation console output:<br>
<img src="./asset/Colorectal_Cancer_evaluate_at_epoch_50_1122.png" width="740" height="auto"><br>
<br>

<br>
Classification report:<br>
<img src="./asset/Colorectanl_Cancer_classificaiton_report_at_epoch_50_1122.png" width="740" height="auto"><br>

<br>
Confusion matrix:<br>
<img src="./projects/Colorectal-Cancer/evaluation/confusion_matrix.png" width="740" height="auto"><br>


<hr>

<h3>
References
</h3>
<b>1.Zenodo: 100,000 histological images of human colorectal cancer and healthy tissue
</b><br>
<pre>
https://zenodo.org/record/1214456#.Y3sVDHZBy3A
</pre>
