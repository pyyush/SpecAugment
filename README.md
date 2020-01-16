## SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition

### Introduction:
This is an implementation of the speech data augmentation method presented in "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition[1]".

##### Base Input
<figure>
  <img src="demo/base.png"/>
</figure>

##### SpecAugment Applied (Time Warp + Frequency Masking + Time Masking)
<figure>
  <img src="demo/time_mask.png"/>
</figure>

### Requirements:
1. python3
2. librosa
3. libsndfile
4. audioread
5. ffmpeg
5. numpy
6. tensorflow
7. tensorflow_addons

### Usage:
```
main.py [--dir][--policy]
```

--dir    | path/to/dataset | default='./LibriSpeech/'\
--policy | augmentation policy to use from {'LB','LD', 'SS', 'SM'} | deafault='LD'

OR

```
refer to demo/demo.ipynb for jupyter notebook demo
```


### References:
1. @article{Park_2019,
   title={SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition},
   url={http://dx.doi.org/10.21437/Interspeech.2019-2680},
   DOI={10.21437/interspeech.2019-2680},
   journal={Interspeech 2019},
   publisher={ISCA},
   author={Park, Daniel S. and Chan, William and Zhang, Yu and Chiu, Chung-Cheng and Zoph, Barret and Cubuk, Ekin D. and Le, Quoc V.},
   year={2019},
   month={Sep}
}
