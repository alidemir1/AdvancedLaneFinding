# Advanced Lane Line Finding

This project is created for Lane Finding
Processed Video can be seen [here](https://www.youtube.com/watch?v=41AABeyhBKg)



# Introduction
The steps of this project are as follows:  
 
* Apply Inverse Perspective Mapping (IPM) to obtain lane lines as parallels.
* Use different filters to obtain line features. 
* Locate lines belongs to the related lane.
* Using location information fit a 2nd degree polynomials to the lines.
* Visualize lane on the IPMed image.
* Warp the IPMed image back to the original.
* Find Radius of the Curvature of Road and Offset of the vehicle from center of the lane
* Run the entire pipeline on a sample [video](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/project_video.mp4). 

---
[//]: # (Image References)

[image1]: ./outputImages/input.PNG "Input Image"
[image2]: ./outputImages/IPMed.PNG "IPM Image"
[image3]: ./outputImages/IPMfiltered.PNG "IPM Filtered"
[image4]: ./outputImages/laneFinding.PNG "Lane Finding"
[image5]: ./outputImages/output.PNG "Output"


# Test Image Pipeline
I created my own pipeline mainly as follows. There can be little differences in the code. Since I am not giving all operations here (e.g. operations in order to reduce the noise, or how to estimate other line when there is only one of them detected). Input image is illustrated as follows.
I created and implemented this pipeline over a night, so if there are any catastrophic errors that you observe please let me know. Pipeline is still prone to the noises such as fast light changes, or too noisy backgrounds for applied filters (which are mentioned below).

![Input Image][image1]

## Inverse Perspective Mapping (IPM)
IPM is used to obtain the bird eye view. To obtain this getPerspective and warpPerspective operators of the Opencv library is used. Since it gathers line as parallels, it eases some operations. For example: If you find one line feature in the bottom of the image, IPM lets you search for features around same x coordinate, while only changing y coordinates.
IPMed version of the Input image is given:

![IPM Image][image2]
 

## Identifying Lane Lines
In order to find the lines, IPMed image is converted into grayscale image. Then Canny operator of the OpenCV library is used. Sobel operator on the X direction is applied and 
following Sobel operator, Morphology Closing is used to fill the lines which their borders are detected. Filtered image is shown below:

![IPM Filtered][image3]
 
To locate the lines in the filtered image, first I look to the below (near to the car) of the image since occlusion chances (e.g. by other vehicle) are low. I created some band with specified height at bottom of the filtered image.
I divide band in to two section Left and Right from the center of the band. Summed all values over the Y direction and find indices of the maximum values which belongs to the left and right band. Map this indices back onto the filtered image. 
Then I looked at the upper of the lane indices found from band with some margin on the X direction, then followed this procedure until the top of the image. Note that indice margin referenced to the previous line indice found on the same side (left or right), references from band only used to find the neighbour upper indices.
Lastly using, this indices which belongs to left and right lines. 2nd degree polynomial is fitted onto IPMed image.

![Lane Finding][image4]



## Radius of the Curvature of the Road and Offset of the Vehicle From Center of the Lane

The radius of curvature is computed as follows; for y = f(x), the radius of curvature is calculated by R = [(1+(dy/dx)^2 )^3/2]/|d^2y / dx^2|.

The offset from the center of the lane is computed by pixel difference from the width/2 of the image from the lane center which can be easily calculated by (Ypoly(image height - 1) - LeftPoly(image height - 1))/2 + LeftPoly(image height - 1) where LeftPoly and RightPoly are polynomials fitted on the left and right lanes respectively and 'image height - 1' is the input Y value (indice of the bottom row). 
In order to have these values in meters meter/pixel values should be estimated and new polynomials should be fitted with those values (for more see the code).
 

## Warping IPMed Image Back to the Original



![Output][image5]
 
