# Action_recognition #
This is an implementation of action recognition algorithm based on scene. The algorithm utilizes the *Inception v3* model and the *LSTM* model and is implemented using the *tensorflow* platform. This algorithm is suitable for video containing individual actions. **(Because it is an early implementation, the project may contain Chinese)**  
The overall process of the algorithm is as follows:  
<div align=center><img src="framework.png"></div>

The process description: firstly, a video with variable length is sampled with the same number of frame images at different frame intervals to form the image sequence group of video.Each image is extracted by the Inception v3 convolutional neural network trained on the ImageNet dataset to obtain a feature sequence group for each video.Then input the feature sequence group of fixed length into LSTM network in time order, extract the spatio-temporal features of the sequence, and complete the action classification task through the full connection layer and the softmax layer.  

## Dependencies ##
python >= 3.5  
tensorflow >= 1.12  
In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:  
- `glob`  
- `tqdm`

Also, you need to install `ffmpeg` to process video.

## Data Preprocessing ##
Download the [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) dataset. Then extract them to `data/`.  
Download the [Inception_v3](https://pan.baidu.com/s/1X8BpCssc1SwCYa7Lkn4UzQ) model pretrained on ImageNet with the verification code **ej0f** to `inception_v3/`.  
You need to execute the following steps successively to implement the data preprocessing:  
- In `data/` directory, you need to run `move_files.py` to split dataset into train and test, and move them to the appropriate folder.  
- Then, in the same directory, you need to run `extract_files.py` to extract useful video information, such as video name, label, number of frames and so on.  
- finally, in the project root directory, you need to run `extract_cnn_features.py` to extract cnn_feature of each images.

## Training ##
You can adjust the parameters to specify the categories to be trained, and if you have sufficient computing resources, ignore the parameters for this problem. You can simply run `train_lstm.py` to train the model.  

## Test ##  
You can simply run `demo.py` to test your model. By the way, you should select a model file to initialize model parameters. You can simply assign the path of the model file to the corresponding parameter.  

## Result ##
I randomly selected 10 classes in UCF101 dataset for training. Using the test dataset to test the model, the accuracy of video action prediction can reach more than 90%.
