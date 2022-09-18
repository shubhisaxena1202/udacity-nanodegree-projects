# Capstone Project : Climate Anomaly Detection of Satellite Images

## Installations <br>
1. Numpy
2. Pandas
3. Seaborn
4. Matplotlib
5. Sklearn
6. os

No additional installations required beyond Anaconda distribution of Python and Jupyter Notebooks 

## Overview & Motivation <br>
Anomaly detection is the process of identifying unexpected entities or trends in data. A relatively new form of anomaly detection is the anomaly detection of satellite imagery. The purpose of this project was to use unsupervised machine learning techniques to find outliers in an image dataset assembled from 1667 Landsat- 8 images from the USGS Earth Explorer. <br>
The images were a collection of high surface reflectance and low cloud cover images from twelve locations in California from 2014 to 2020.<br>
Moreover,in this project the Landsat-8 images were compressed using dimensionality reduction techniques like Principal Component Analysis (PCA) and Incremental PCA. A sample of reduced images were then classified into normal and anomalous images and further divided into training and test set partitions.

### KMeans Clustering for Anomaly Detection <br>
Keeping in mind the complexity of Udacity's nanodegree program, only KMeans clustering algorithm is performed to determine if land satellite images have anomalies or not. K-means clustering is an unsupervised machine learning technique in which the number of k-clusters are chosen prior to the clustering which also corresponds to the number of centroids for the image data.

### Principal Component Analysis for Dimensionality Reduction <br>
Principal Component Analysis or PCA was a dimensionality reduction technique used in order to reduce the dimensions of the Landsat-8 images. PCA transforms the features of data by combining them into uncorrelated linear combinations. The image can be reconstructed using the number of components needed to explain the threshold variance.

## Data Acquistion <br>
The image dataset was created by querying USGS Earth Explorer for Landsat-8 images in California. The cri- teria to fetch included geospatial coordinates, date range, surface reflectance and cloud cover. Moreover, the images were filtered to have high surface reflectance and cloud cover ranging from 0 to 10 percent. <br>
Please contact the owner of this repository to have access to all images downloaded from the USGS Earth Explorer. <br>
For simplicity a few images have been added to the 'data' folder of this project.

## File Descriptions <br>
    1. Data Folder : 
        a. 34 : 
            Original Sample images of location 34 over the specified time frame. 
        b. 34_cropped : 
            Cropped images of location 34 as specified above, without the watermark. This is the output folder of preprocess.py script in the project.
        c. test :
           This folder contains sample images that have been used for test purposes of algorithms currently beyond the scope of this project, keeping in mind Udacity's complexity.
        d. train : 
           This folder contains sample images that have been used for test purposes of algorithms currently beyond the scope of this project, keeping in mind Udacity's complexity.
    2.  k-means.ipynb :
        This notebook contains source code for performing KMeans algorithm for 1 sample image to reduce complexity.
    3.  pcaimage.ipynb :
        This notebook contains the source code for performing PCA and Incremental PCA to reconstruct images.
    4.  preprocess.py :
        This python script is used to create training & testing sample of images from raw image dataset by removing the watermark from the images.

## Licensing, Authors, and Acknowledgements
Rosebrock, A. 2020. “Intro to Anomaly Detection with OpenCV, Computer Vision, and Scikit-Learn.” PyImageSearch, www.pyimagesearch.com/2020/01/20/intro-to-anomaly-detection-with-opencv-computer-vision- and-scikit-learn/.

## Link to Medium Blog Post:
https://medium.com/@shubhiS/climate-anomaly-detection-of-satellite-images-ee0c8e644bbb


