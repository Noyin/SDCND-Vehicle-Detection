
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images 
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2a]: ./output_images/HOG_example_1.jpg
[image2b]: ./output_images/HOG_example_2.jpg
[image2c]: ./output_images/HOG_example_3.jpg
[image2d]: ./output_images/HOG_example_4.jpg
[image3]: ./colorspace_exploration/HSV_3D_plot.png
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

The code for the following steps are contained in the IPython notebook located in "./CarND-Vehicle-Detection.ipynb". 

**Histogram of Oriented Gradients (HOG)**

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In code cell 2 of the Ipython notebook , I read in the provided the `vehicle` and `non-vehicle` images into cars and not_cars arrays respectively.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

In code cell 5, I explored various color spaces of a randomly selected car and non car image using `explore_colorspaces_HOG` function and computed the HOG of each colorspace using `skimage.hog()` function with  different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I displayed the images to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2a]
![alt text][image2b]
![alt text][image2c]
![alt text][image2d]

I tried various combinations of parameters and selected the following combination `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. This combination properly captured unique features of vehicles in images. 

**Color Histograms**

From code cell 6 - 12, I explored various color histograms of colorspaces to determine which properly clusters car pixels.The Saruration plane in the SHSV color sapce does a good good.Here is a 3-D plot of an image show the clustering of car pixels in the saturation plane:

![alt text][image3]




**Train Classifier**

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  







**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/test6.jpg "Road Transformed"
[image3]: ./output_images/binary_combo_example.jpg "Binary Example"
[image4]: ./output_images/ROI_image.jpg "ROI" 
[image5a]: ./output_images/warp_confirmation.jpg "Warp Example"
[image5b]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image6]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image7]: ./output_images/example_output.jpg "Output" 
[image8]: ./processed_project_video.gif "Video"
[video]: ./processed_project_video.mp4 "Video"


The code for the following steps are  contained in the IPython notebook located in "./CarND-Advanced-Lane-Lines .ipynb". 

### Camera Calibration


In code cell 2, I calculated camera matrix and distortion coefficients using the `calculate_camera_matrix_and_disturtion_coefficients()` function. I first of all created an object point `objp` which represents a chess board fixed on the x,y plane with z = 0.Then I looped through the calibration images and applied `cv2.findChessboardCorners()` function on each image. Whenerver corners are detected I saved a copy of an object point `objp` to an object points array `objpoints` and saved the detected corners of an image to an image points array `imgpoints`.I passed the object points array `objpoints` and image points array `imgpoints` as inputs to `cv2.calibrateCamera()` function which returns camera matrix `mtx` and distortion coefficients `dist`. I applied the camera matrix `mtx` and distortion coefficients `dist` to an image using `cv2.undistort()` function and got the following result:
![alt text][image1]

### Distortion correction
In code cell 3, I applied the calculated camera matrix `mtx` and distortion coefficients `dist` to test images using `cv2.undistort()` function and below is the result on a test image:
![alt text][image2]

### Application of color transforms and gradients

From code cells 4 to 19 , I explored various colour channels and calculated sobel in the horizontal direction , sobel in the vertical direction using the `cv2.Sobel()` function and then  calculated the magnitude and direction of their gradient. Of all the colour channels, I choose L channel from the HSL color space as it performs well under changing lighting conditons and S channel from the HSV colour space as it generally perfomes well compared to other channels in identifying lane lines. I applied a threshold to the outputted binary image with minimum and maximum values set to 10 and 150 respectively.

I combined the thresholded binary image from the L channel and S channel to get the following result:

![alt text][image3]

I also applied a `isolate_region_interest()` function in code cell 20,  to isolate only lane lines in the thresholded binary image. Below is an example:

![alt text][image4]

### Perspective transform

I derived perspective tranform `M` in code cell 21, by using `cv2.getPerspectiveTransform()` function which takes in an array of source points `src` and destination points `dst`. Below are the harcoded values used for source points `src` and destination points `dst`:

```python
src = np.float32([(257, 685), (1100, 685), (583, 460),(720, 460)])
dst = np.float32([(200, 720), (1080, 720), (200, 0), (1080, 0)])
```

I then warped an image using `cv2.warpPerspective()` function , which takes in an image and perspective tranform `M` as inputs. To verify that my perspective transform was working as expected I drew the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.Below is the result:
![alt text][image5a]


Also the binary output of the warped image is shown below:

![alt text][image5b]


### Detect lane pixels and fitting with a polynomial

In code cell 24 , I applied a sliding window search using `find_window_centroids()` function and passing in a warped image as input. I then used `window_centers()` function to fit the detected lane pixels with a polynomial. Below is an example:
![alt text][image6]


### Calculate radius of curvature and position of vehicle with respect to center

In code cell 25 , I used  `pos_from_center()` function to calculate the center position of vehicle with respect to center and `get_curvature()` function to calculate radius of curvature of the lane lines.


### Final image with identified lane area

In code cell 27, I used `create_overlay()` function to create an overlay and performed a perspective tranform using `cv2.getPerspectiveTransform()` function with an inverse perspective tranform `Minv` as input.The resulting overlay covers the identified lane area. Here is an example of applying the `create_overlay()` function on a test image:
![alt text][image7]

---

### Pipeline (video)

In code cell 29 , I created a pipeline to output an image with an overlay showing detected lane region and text showing lane curvature and vehicle center position.I passed each frame of the project video through the pipeline and obtained the result below:

![alt text][image8]

Here's a [link to the processed video][video]

---

### Discussion
The output from the pipeline generally performed well on straight roads and slight curves. It also perfomed well under different lighting conditions.
Though, the pipeline performs poorly on sharp curves, in situations where only one lane line is visible in the image(frame). Also its fails to differentiate between dark lines(edge of road or newly tarred sections of the road) and lane lines.It performs poorly on inclined roads.

To make a more robust pipeline, I would improve the thresholded binary image to identify only lanes lines (i.e differentiate between dark edges and lane lines). I would also improve the pipeline to be able to identify single lane lines in an image(frame). I would implement an adjustable second order polynomial fitting of lane lines for different road scenarios such as flat and inclined roads.





