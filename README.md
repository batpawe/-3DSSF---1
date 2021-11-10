
# 3DSSF - 1st assignment

This repository contains different algorithms about stereo vision matching, displaying and analysing. In particular C++ implementation contains features like:

 - Reading and saving images using OpenCV,
 - Disparity image generated by naive stereo matching using **SSD technique**,
 - Disparity image generated by **dynamic programming** approach,
 - Different metrics (**SSIM, Peak signal to noise ratio**),
 - Pointcloud generator for disparity images and writer to **.ply file**,
 - **OpenMP** implementation of SSD and DP stereo matching,

All 6 stereo pair images were taken from [Middlebury stereo datasets](http://vision.middlebury.edu/stereo/data) together with ground truth image provided in the package.
Different parameters (in pointcloud generator) were taken from Middlebury as well. *(_3740px_ and _160mm_ for the focal length and the baseline).*

# How to run the code
In order to run the code you must generate binary file using cmake. CMakeLists.txt is provided in the repository. 

Beside that to run the code user has to have OpenCV library for C++.

# Results
Below you can find results for different Middlebury stereo pairs. All of the images, 3d models and other information can be found in the data folder (for each set seperately).

If the images are not showing up please contact me through github. (it is possible that the host of the images deleted them)

## Stereo pairs

6 different stereo pair images presented below were used for my analysis. These images differ in size, therefore it will be visible in the analysis that some disparity images are calculated much faster than the others.

For each disparity image .ply file was generated and then visualized in Meshlab software. Screenshots of these 3d models can be found below. .ply files can be found in this repository as well.

For all images 4 disparity images were created:

 - **SSD**(patch_size = 2, disparity = 4)
 - **SSD**(patch_size = 12, disparity = 14)
 - **DP**(occlusion = 18)
 - **StereoBM (OpenCV)**(disparity = 16, patch_size = 5)

The reason for using same parameters in all cases is to show that there is are no gold parameters that will work with all images. Each image has to be taken individually and therefore some results might not be satisfactory.

![Image stereo pairs](https://i.ibb.co/nQXkm3p/combine-images.jpg)
## Image set 1 - Art

![Side by side comparison](https://i.ibb.co/dbVJqKN/Combined-images.jpg)

Image size: 1390x1110
Statistics for each algorithm:
| Algorihm | Time | OpenMP Time | Signal to Noise ratio | SSIM Index |
|--|--|--|--|--|
| SSD(2,4) | <1second |<1 second | 28,24 | 6417 |
| SSD(12,14) | 15 seconds |2 seconds | 26,15 | 7287 |
| DP(18) | 76 seconds | 12 seconds | 33,89 | 6210 |
| OpenCV StereoBM(16, 5) | 0 seconds | - | 26.53 | - |

### OpenCM StereoBM result

![Stereo BM OpenCV](https://i.ibb.co/VWdQq5t/stereo-BM-left.png)


### Pointcloud 3d model
![3D pointcloud model](https://i.ibb.co/JrxmHzT/img.png)

**Conclusion**
In case of "Art" stereo pair dynamic programming creates much more comprehensive disparity image than in case of SSD. The disadvantage of that is that DP costs more time. In terms of scores all of the images are more or less the same.

## Image set 2 - Masks
![Side by side comparison of the results](https://i.ibb.co/hK9tgmK/combine-images.jpg)

Image size: 450x375
Statistics for each algorithm:
| Algorihm | Time | OpenMP | Signal to Noise ratio | SSIM Index |
|--|--|--|--|--|
| SSD(2,4) | <1second | <0 second | 28,35 | 51 |
| SSD(12,14) | 1 second |<0 second | 26,78 | 92 |
| DP(18) | 2 seconds |<0 second | 33,63 | 184 |
| OpenCV StereoBM(16, 5) | 0 seconds | 26,83 | 26.83 | - |

### OpenCM StereoBM result

![Stereo BM OpenCV](https://i.ibb.co/jkcD1rs/stereo-BM-left.png)


### Pointcloud 3d model
![3D Pointcloud model](https://i.ibb.co/2KmKfrt/img.png)

**Conclusion**
In case of "Mask" stereo pair dynamic programming creates much more comprehensive disparity image than in case of SSD, here to solve the problem of SSDs results it would be nice to tweak a little bit the parameters.

## Image set 3 - Toys
![Side by side comparison](https://i.ibb.co/KwR64X0/combine-images.jpg)

Image size: 450x375
Statistics for each algorithm:
| Algorihm | Time | OpenMP Time | Signal to Noise ratio | SSIM Index |
|--|--|--|--|--|
| SSD(2,4) | <1second |<0 second | 28,16 | 0 |
| SSD(12,14) | 1 second |<0 second | 26,44 | 0 |
| DP(18) | 2 seconds |<0 second | 34,43 | 0 |
| OpenCV StereoBM(16, 5) | 0 seconds | 26,93 | 26.93 | - |

### OpenCM StereoBM result

![StereoBM Results](https://i.ibb.co/JCRSwJ4/stereo-BM-left.png)


### Pointcloud 3d model
![3D Pointcloud model](https://i.ibb.co/JrF1ZK0/img.png)

**Conclusion**
In case of "Toys" stereo pair dynamic programming creates much more comprehensive disparity image than in case of SSD, here to solve the problem of SSDs results it would be nice to tweak a little bit the parameters just as in previous pair. What's more it seems that for small size images the Signal to Noise ratio and SSIM Index show meaningless information.

## Image set 4 - Plant

![Side by side comparison](https://i.ibb.co/SKV87DY/combine-images.jpg)
Image size: 1282x1110
Statistics for each algorithm:
| Algorihm | Time | OpenMP | Signal to Noise ratio | SSIM Index |
|--|--|--|--|--|
| SSD(2,4) | <1second | <0 second | 28,35 | 1180 |
| SSD(12,14) | 14 seconds | 2 seconds | 25,21 | 193 |
| DP(18) | 67 secondss | 10 seconds | 35,81 | 615 |
| OpenCV StereoBM(16, 5) | 2 seconds |  -| 25.21 | - |

### OpenCM StereoBM result

![Stereo BM Results](https://i.ibb.co/wwyhV28/stereo-BM-left.png)

### Pointcloud 3d model
![3D Pointcloud model](https://i.ibb.co/pX1DBnk/img.png)**Conclusion**
Seems that with "easier" shapes or rather single objects on the image, naive SSD works better than in case of more complicated images (i.e. SSIM score for SSD is better than for DP)

## Image set 5 - Bowl
![Side by side comparison](https://i.ibb.co/PthTKK1/combine-images.jpg)
Image size: 1252x1110
Statistics for each algorithm:
| Algorihm | Time |OpenMP| Signal to Noise ratio | SSIM Index |
|--|--|--|--|--|
| SSD(2,4) | <1second |0 second | 22,34 | 346 |
| SSD(12,14) | 14 seconds |2 seconds | 26,12 | 95 |
| DP(18) | 64 seconds |10 seconds | 18,09 | 193 |
| OpenCV StereoBM(16, 5) | 2 seconds |  -| 26,37 | - |

### OpenCM StereoBM result

![Stereo BM OpenCV](https://i.ibb.co/2NZ37Pk/stereo-BM-left.png)

### Pointcloud 3d model
![3D Pointcloud model](https://i.ibb.co/jrXRbKm/img.png)

**Conclusion**
Here once again Naive algorithm shows quite nice results with good score and disparity image.

## Image set 6 - Bowl
![Side by side comparison](https://i.ibb.co/dKVbGzm/combine-images.jpg)

Image size: 1312x1110
Statistics for each algorithm:
| Algorihm | Time |OpenMP| Signal to Noise ratio | SSIM Index |
|--|--|--|--|--|
| SSD(2,4) | <1second |0 second | 22,34 | 19 |
| SSD(12,14) | 15 seconds |2 seconds | 26,12 | 7 |
| DP(18) | 71 seconds |11 seconds | 18,09 | 4294967326 |
| OpenCV StereoBM(16, 5) | 2 seconds |  -| 25,51 | - |

### OpenCM StereoBM result

![Stereo BM OpenCV](https://i.ibb.co/0hjx22W/stereo-BM-left.png)

### Pointcloud 3d model

![3D Pointcloud model](https://i.ibb.co/rQvB9k4/img.png)

**Conclusion**
Case of pots is quite unusual because for DP algo, we can see that the SSIM score is super high. However (I did not check that), I think it is because of some memory issue in the code.

## General conclusions

Some general conclusion that I can see after analysing these 6 sets.

 1. Dynamic Programming approach creates more comprehensive disparity images (especially on complex images). However it does this with cost of time. Usually DP was couple times longer than naive approach.
 2. Naive SSD creates great results for simple pairs but it has to be well parametrized. Much more than DP approach.
 3. ~~There are possible improvements that can be done. I.e. OpenMP library could be used to boost the time cost of the algortihms.~~
 4. The total time of analysing these 6 stereo pairs was **392 seconds** - without OpenMP and **100 seconds** with OpenMP.
 5. Generating 3D pointcloud can be improved for sure (with tweaking the camera parameters) but I think mine results are quite satisfactory.
 6. Bigger images (greater size) takes much more time to compute than smaller images.
 7. StereoBM from OpenCV is as fast as my implementation of SSD (in OpenMP) but in my opinion results are not always satisfactory. I prefer to use SSD which may work a little bit slower but gives better results.

## Optionals done

### OpenMP
I implemented OpenMP library to the code so I could see how it will improve the way the algorithms works.
Of course OpenMP pragmas were written in such a place that it won't change the result of the disparity images. These places can be found in code (it should be easily visible in last commit).

Using OpenMP I noticed more ten times faster performance of all stereo pair. Making the longest DP calculation faster than SSD(12,14) for no OpenMP implementation.
### StereoBM from OpenCV
In the code I used StereoBM disparity image algorithm provided by OpenCV to compute disparity maps and compare scores achieved by my metrics.
