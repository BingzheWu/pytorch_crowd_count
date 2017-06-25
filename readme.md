## High Density Crowd Counting
 
This is a technique used to count or estimate the number of people in a crowd.

Our implementation is based on CrowdNet, but we made some changes.

You can use the code to estimate the density(actually the number of the people in a static picture) of people.

### Disclaimer
This is a re-implementation of original CrowdNet.
The official repository is  available [here](https://github.com/davideverona/deep-crowd-counting_crowdnet)
The arxiv paper is available [here](https://arxiv.org/abs/1608.06197.pdf)

### Getting started
You need python modules: 'cv2', 'torch', 'numpy', 'h5py', 'matplotlib', etc..
``` 
	sudo apt-get install python-opencv python-matplotlib python-numpy
	Installing the pytorch may be a little more complex, you could find the details [here](https://github.com/pytorch/pytorch#installation) 
    
```

Remember to enable the CUDA, since it is very slow using cpu.

### Before training

* python data_process.py
in this part, we load the img and the ground truth， do gaussian filter to the ground truth(with KD-tree) , split the pictures into pieces and store them in the document ' dataset/processed_hdf5' (surely we store the patches in the form of h5)

* python loader.py
in this part, we load the patches in the 'dataset/processed_hdf5' to torch. we load 20 patches one time.

### Train

* python train.py
train the net in this part, until there is no patch in that document.
the net trained is stored in 'checkpoint/crowd_net19.pth'

### Try or Evalute
* python demo.py
you put a picture into the demo, and it will return a heatmap for you. if you want to find out how many people in this picture, just do integration.

#### To be continue...


    
