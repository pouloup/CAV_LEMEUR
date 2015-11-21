#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

void conv2(vector<Mat> &data, vector<Mat> &bl_data, int kernel_size);
void pickROI(vector<Mat> &data, vector<Mat> &bl_data, vector<Mat> &HR_dic, vector<Mat> &LR_dic);

int main(int argc, char** argv){

  cv::String path("../data/training/*.bmp"); //select only bmp.

  vector<cv::String> fn;
  vector<Mat> data, bl_data; //Blurred HR Images.
  vector<Mat> HR_dic, LR_dic;

  cv::glob(path,fn,true); // recurse.

  for (size_t k=0; k<fn.size(); ++k){
    Mat im = imread(fn[k]/*, IMREAD_GRAYSCALE*/);
    if( !im.data ) cout <<  "Could not open or find an image" << endl;  // Check for invalid input.
    data.push_back(im);
  }

  //Blurring the HR images.
  conv2(data, bl_data, 3);
  if(data.size() != bl_data.size()) cerr << "Error during convolution : number of images is irrelevant.";

  pickROI(data, bl_data, HR_dic, LR_dic);

  return 0;
}

void conv2(vector<cv::Mat> &data, vector<cv::Mat> &bl_data, int kernel_size){
	Mat dst, kernel;
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);

	/// Apply filter
	for (unsigned int i = 0; i < data.size(); ++i){
		filter2D(data[i], dst, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
		bl_data.push_back(dst);
	}
}

void pickROI(vector<Mat> &data, vector<Mat> &bl_data, vector<Mat> &HR_dic, vector<Mat> &LR_dic){

	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 200;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 5;
  /*
  1  = 157 area detected
  3  = 74  area detected
  5  = 43  area detected
  7  = 18  area detected
  10 = 6   area detected
  */


	// Set up the detector with default parameters.
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

  for(int i = 0; i < data.size(); ++i){
  	// Detect blobs of important area in not blurred picture i.
  	vector<KeyPoint> keypoints;
  	detector->detect(data[i], keypoints);

    /*
      retrieve a cv::Size patch with the keypoints as center.
      and then create dictionaries with similar patches but one is from the HR images
      and the other is from the same images but blurred.
    */

    for(int j = 0; j < keypoints.size(); ++j){
      Mat patch_HR, patch_HR_bl;
      getRectSubPix(data[i], cv::Size(5,5), keypoints[j].pt, patch_HR);
      getRectSubPix(bl_data[i], cv::Size(5,5), keypoints[j].pt, patch_HR_bl);

      HR_dic.push_back(patch_HR);
      LR_dic.push_back(patch_HR_bl);
    }
  }

	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
  /*
  Mat im_with_keypoints;
	drawKeypoints(data[0], keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Show blobs
	imshow("keypoints", im_with_keypoints);
	waitKey(0);
  */
}
