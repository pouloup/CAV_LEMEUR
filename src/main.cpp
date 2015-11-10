#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

void pickROI(vector<cv::Mat> &data, vector<cv::Mat> &bl_data){
  
}

void conv2(vector<cv::Mat> &data, vector<cv::Mat> &bl_data, int kernel_size)
{
    Mat dst, kernel;
    kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
    //namedWindow( "test", CV_WINDOW_AUTOSIZE );
    /// Apply filter
    for(int i = 0; i < data.size(); ++i){
      //imshow( "test", data[i] );
      //waitKey(0);
      filter2D(data[i], dst, -1 , kernel, Point( -1, -1 ), 0, BORDER_DEFAULT );
      bl_data.push_back(dst);
      //imshow( "test", dst );
      //waitKey(0);
    }
}

int main(int argc, char** argv){

  cv::String path("../data/training/*.bmp"); //select only bmp.

  vector<cv::String> fn;
  vector<Mat> data; //HR Images.
  vector<Mat> bl_data; //Blurred HR Images.

  cv::glob(path,fn,true); // recurse.

  for (size_t k=0; k<fn.size(); ++k){
    Mat im = imread(fn[k]);
    if( !im.data ) cout <<  "Could not open or find an image" << endl;  // Check for invalid input.
    data.push_back(im);
  }

  //Blurring the HR images.
  conv2(data, bl_data, 3);
  if(data.size() != bl_data.size()) cerr << "Error during convolution : no such same number of images.";

  return 0;
}
