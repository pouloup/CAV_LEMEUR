#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

/*
 * TODO : Pick les ROI pour toutes les images, prendre les 15 premiers points par ex et matcher les deux dictionnaires.
 */
void pickROI(vector<cv::Mat> &data, vector<cv::Mat> &bl_data){

  cv::Ptr< cv::Feature2D > detector = cv::xfeatures2d::SIFT::create(0,numScales,peakThresh,edgeThresh,sigmaScale);

  // Extraction de key points
  std::vector< cv::KeyPoint> keyPts;
  detector->detect(img,keyPts);

  //Flitrage des key points
  // Je te conseil d'utiliser des filtres sur tes keyPoints histoire de bien adapter des données à ton application
  cv::KeyPointsFilter filter;
  if( mConfig.keyPointSizeMax > 0.0f )
  {
     double sizeMin = 2.0 ; // J'enleve les keyPoint dont l'echelle est inferieur à 2 pixels. Il y a d'autres filtres possibles...
     filter.runByKeypointSize(keyPts,sizeMin);
  }

  // extraction des descripteurs
  cv::Mat descriptors;
  detector->compute(img,keyPts,descriptors);
}

void conv2(vector<cv::Mat> &data, vector<cv::Mat> &bl_data, int kernel_size)
{
    Mat dst, kernel;
    kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);

    /// Apply filter
    for(int i = 0; i < data.size(); ++i){
      filter2D(data[i], dst, -1 , kernel, Point( -1, -1 ), 0, BORDER_DEFAULT );
      bl_data.push_back(dst);
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
