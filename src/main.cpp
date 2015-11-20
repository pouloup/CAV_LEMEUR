#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

void conv2(vector<cv::Mat> &data, vector<cv::Mat> &bl_data, int kernel_size);
void pickROI(vector<cv::Mat> &data, vector<cv::Mat> &bl_data);

int main(int argc, char** argv){

  cv::String path("../data/training/*.bmp"); //select only bmp.

  vector<cv::String> fn;
  vector<Mat> data; //HR Images.
  vector<Mat> bl_data; //Blurred HR Images.

  cv::glob(path,fn,true); // recurse.

  for (size_t k=0; k<fn.size(); ++k){
    Mat im = imread(fn[k]/*, IMREAD_GRAYSCALE*/);
    if( !im.data ) cout <<  "Could not open or find an image" << endl;  // Check for invalid input.
    data.push_back(im);
  }

  //Blurring the HR images.
  conv2(data, bl_data, 3);
  if(data.size() != bl_data.size()) cerr << "Error during convolution : no such same number of images.";

  pickROI(data, bl_data);

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

/*
* TODO : Pick les ROI pour toutes les images, Harris blobs detection.
*/
void pickROI(vector<cv::Mat> &data, vector<cv::Mat> &bl_data){
	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 200;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 3;
  /*
  1  = 157 area detected
  3  = 74  area detected
  5  = 43  area detected
  7  = 18  area detected
  10 = 6   area detected
  */


	// Set up the detector with default parameters.

	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	// SimpleBlobDetector::create creates a smart pointer.
	// So you need to use arrow ( ->) instead of dot ( . )
	// detector->detect( im, keypoints);

	// Detect blobs.
	std::vector<KeyPoint> keypoints;
	detector->detect(data[0], keypoints);

  std::cout << keypoints.size() << std::endl;

	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
	Mat im_with_keypoints;
	drawKeypoints(data[0], keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Show blobs
	imshow("keypoints", im_with_keypoints);
	waitKey(0);
}
