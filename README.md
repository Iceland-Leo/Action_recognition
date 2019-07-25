# Action_recognition #
This is an implementation of action recognition algorithm based on scene. The algorithm utilizes the *Inception v3* model and the *LSTM* model and is implemented using the *tensorflow* platform. This algorithm is suitable for video containing individual actions. **(Because it is an early implementation, the project may contain Chinese)**  
The overall process of the algorithm is as follows:  
<div align=center><img src="framework.png"></div>
Firstly, video with variable length is sampled with the same number of frame images at different frame intervals to form the image sequence group of video.Each image is extracted by the Inception v3 convolutional neural network trained on the ImageNet dataset to obtain a feature sequence group for each video.Then input the feature sequence group of fixed length into LSTM network in time order, extract the spatio-temporal features of the sequence, and complete the action classification task through the full connection layer and the softmax layer.  

## Dependencies ##
