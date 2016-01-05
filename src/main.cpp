#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>
#include <vector>
#include <tuple>

using namespace std;
using namespace cv;

typedef vector<std::tuple<Mat, Mat, Mat, Mat>> vTuple;
typedef vector<cv::Mat> vMat;
typedef std::tuple<Mat, Mat, Mat, Mat> tMat;

vTuple HR_dic, LR_dic;

void superResolution();
bool loadDic();
bool constructionDictionnaires();
void conv2(vMat &data, vMat &bl_data, int kernel_size);
void pickROI(vMat &data, vMat &bl_data, vTuple &HR_dic, vTuple &LR_dic);

int main(int argc, char** argv) {
	
	int choix = 0;

	while (true){
		cout << "__ Veuillez choisir entre : \n __ la construction des dictionnaires (1) \n __ la super résolution d'une image (2)" << endl;
		cin >> choix;

		switch (choix){
		case 1:
			if (constructionDictionnaires())
				cout << "SUCCESS" << endl;
			else
				cout << "FAILURE" << endl;
			break;

		case 2:
			if (HR_dic.size() > 0 && LR_dic.size() > 0)
				superResolution();
			else {
				loadDic();
				superResolution();
			}
			break;

		default:
			break;
		}
	}
}

void superResolution() {

}

bool loadDic() {
	return constructionDictionnaires();
}

bool constructionDictionnaires(){

  cv::String path("../data/*.bmp"); //select only bmp.

  vector<cv::String> fn;
  vMat data, bl_data; //Blurred HR Images.

  glob(path,fn,false); // recurse.
  
  for (unsigned int k=0; k<fn.size(); ++k){
    Mat im = imread(fn[k]);
    if( !im.data ) cout <<  "Could not open or find an image" << endl;  // Check for invalid input.
    data.push_back(im);
  }

  //Blurring the HR images.
  conv2(data, bl_data, 3);
  if(data.size() != bl_data.size()) cerr << "Error during convolution : number of images is irrelevant.";

  pickROI(data, bl_data, HR_dic, LR_dic);

  return HR_dic.size() == LR_dic.size();
}

void conv2(vMat &data, vMat &bl_data, int kernel_size){
	Mat dst, kernel;
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);

	/// Apply filter
	for (unsigned int i = 0; i < data.size(); ++i){
		filter2D(data[i], dst, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
		bl_data.push_back(dst);
	}
}

void pickROI(vMat &data, vMat &bl_data, vTuple &HR_dic, vTuple &LR_dic){

  Mat patch_HR, patch_HR_bl;
  Mat grad_x, grad_y, grad_bl_x, grad_bl_y;
  Mat Lab, Lab_bl;
  tMat HR, HR_bl;

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

  for(unsigned int i = 0; i < data.size(); ++i){
  	// Detect blobs of important area in not blurred picture i.
  	vector<KeyPoint> keypoints;
  	detector->detect(data[i], keypoints);
    /*
    Mat im_with_keypoints;
    drawKeypoints( data[i], keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    imshow("keypoints", im_with_keypoints );
    waitKey(0);
    */
    /*
      retrieve a cv::Size patch with the keypoints as center.
      and then create dictionaries with similar patches but one is from the HR images
      and the other is from the same images but blurred.
    */
    for(unsigned int j = 0; j < keypoints.size(); ++j){

      getRectSubPix(data[i], cv::Size(5,5), keypoints[j].pt, patch_HR);
      getRectSubPix(bl_data[i], cv::Size(5,5), keypoints[j].pt, patch_HR_bl);

	  /*
	  Mat im_with_keypoints;
	  drawKeypoints( data[i], keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	  imshow("keypoints", im_with_keypoints );
	  waitKey(0);
	  */

      //X & Y Gradient of ROI(Sobel Derivative)
      Sobel( patch_HR, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
      Sobel( patch_HR, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );

      Sobel( patch_HR_bl, grad_bl_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
      Sobel( patch_HR_bl, grad_bl_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );

      //Lab values of the patch
      cvtColor(patch_HR, Lab, CV_BGR2Lab);
      cvtColor(patch_HR_bl, Lab_bl, CV_BGR2Lab);

      //Construction of each information structure of the dictionarie
      HR = std::make_tuple(patch_HR, Lab, grad_x, grad_y);
      HR_bl = std::make_tuple(patch_HR_bl, Lab_bl, grad_bl_x, grad_bl_y);

      //Construction of the dictionaries
      HR_dic.push_back(HR);
      LR_dic.push_back(HR_bl);
    }
  }
}
