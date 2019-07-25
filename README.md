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

Also, you need to install `ffmpeg` to process vedio.

## Data ##
Download the [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) dataset. Then extract them to `data/`.
