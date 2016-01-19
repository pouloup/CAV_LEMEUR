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

Scalar getMSSIM( Mat& i1, Mat& i2);
int patch_rank_on_ssim(Mat patchBicub);
int patch_rank_on_grad(Mat patchBicub);
int patch_rank_on_Lab(Mat patchBicub);
void superResolution();
bool loadDic();
bool constructionDictionnaires();
void conv2(vMat &data, vMat &bl_data, int kernel_size);
void pickROI(vMat &data, vMat &bl_data, vTuple &HR_dic, vTuple &LR_dic);

int main(int argc, char** argv) {

	int choix = 0;

	while (true){
		cout << "__ Veuillez choisir entre : \n __ la construction des dictionnaires (1) \n __ la super resolution d'une image (2)" << endl;
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

Scalar getMSSIM( Mat& i1, Mat& i2){
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d     = CV_32F;

    Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    Mat mu1, mu2;   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
    return mssim;
}

void superResolution() {
  Mat im_source = imread( "../data/res/source.bmp" );
	Mat im_dest;
	Mat patch;
	int index = -1;

	cv::resize(im_source, im_dest, Size(), 2, 2, INTER_CUBIC );
	Mat im_final(im_dest.rows, im_dest.cols, im_dest.type());

	for(int i = 2 ; i < im_dest.rows ; i+=5)
		for(int j = 2 ; j < im_dest.cols ; j+=5){
			getRectSubPix(im_dest, cv::Size(5,5), Point2f(i,j), patch);
			index = patch_rank_on_ssim(patch);
			im_final.col(j).row(i) = get<0>(HR_dic[index]);
		}

	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
  imshow( "Display window", im_final );
	waitKey(0);
}

int patch_rank_on_ssim(Mat patchBicub){
	double score = std::numeric_limits<double>::max();
	/*
	 * score1 : SSIM sur les patchs
	 */
	vector<Scalar> ssim_scores;
	for (unsigned int i = 0; i < HR_dic.size(); i++)
		ssim_scores.push_back( getMSSIM(get<0>(HR_dic[i]), patchBicub) );

	int index = -1;

	for (unsigned int i = 0; i < ssim_scores.size(); i++)
		if(ssim_scores[i](0)<score){
			index = i; score = ssim_scores[i](0);
		}

	return index;
}

int patch_rank_on_grad(Mat patchBicub){
	/*
	 * score2 : distance euclidienne entre les couples de gradient
	 */

	double score = std::numeric_limits<double>::max();
	Mat grad_x, grad_y;

	//Gradient of bicubic patch
	Sobel( patchBicub, grad_x, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );
	Sobel( patchBicub, grad_y, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );

	vector<double> liste_scoreX, liste_scoreY;

	for (unsigned int i = 0; i < HR_dic.size(); i++) {
		liste_scoreX.push_back( norm(get<2>(HR_dic[i]), grad_x, NORM_L2) );
		liste_scoreY.push_back( norm(get<3>(HR_dic[i]), grad_y, NORM_L2) );
	}

	int index = 0;

	for (unsigned int i = 0; i < liste_scoreX.size(); i++)
		if( liste_scoreX[i]+liste_scoreY[i] < score){
			index = i; score = liste_scoreX[i]+liste_scoreY[i];
		}

	return index;
}

int patch_rank_on_Lab(Mat patchBicub){
	double score = std::numeric_limits<double>::max();
	/*
	 * score3 : Score sur la valeur L des patchs Lab
	 */

	// Use the o-th channel (L)
	int channels[] = { 0 };
	float L_ranges[] = { 0, 100 };
	const float* ranges[] = { L_ranges };
	int h_bins = 50;
	int histSize[] = { h_bins };

	/// Histograms
  MatND hist_bicub;
  MatND hist_dic;

	vector<double> L_scores;

	for (unsigned int i = 0; i < HR_dic.size(); i++) {
	  /// Calculate the histograms for the Lab images
		calcHist( &patchBicub, 1, channels, Mat(), hist_bicub, 2, histSize, ranges, true, false );
	  normalize( hist_bicub, hist_bicub, 0, 1, NORM_MINMAX, -1, Mat() );

	  calcHist( &get<1>(HR_dic[i]), 1, channels, Mat(), hist_dic, 2, histSize, ranges, true, false );
	  normalize( hist_dic, hist_dic, 0, 1, NORM_MINMAX, -1, Mat() );

		//1 = Correlation
		//2 = Chi-Square
		//3 = Intersection
		//4 = Bhattacharyya
		L_scores.push_back( compareHist( hist_bicub, hist_dic, 1 ) );
	}

	int index = 0;

	for (unsigned int i = 0; i < L_scores.size(); i++)
		if( L_scores[i] < score){
			index = i; score = L_scores[i];
		}

	return index;
}

bool loadDic() {
	return constructionDictionnaires();
}

bool constructionDictionnaires(){

  cv::String path("../data/training/*.bmp"); //select only bmp.

  vector<cv::String> fn;
  vMat data, bl_data; //Blurred HR Images.

  glob(path,fn,false); // recurse.

  for (unsigned int k=0; k<fn.size(); ++k){
    Mat im = imread( fn[k] );
    if( !im.data ) cout <<  "Could not open or find an image" << endl;  // Check for invalid input.
    data.push_back(im);
  }

  //Blurring the HR images.
  conv2(data, bl_data, 3);
  if(data.size() != bl_data.size()) cerr << "Error during convolution : number of images is irrelevant.";

  pickROI(data, bl_data, HR_dic, LR_dic);

	std::cout << HR_dic.size() << "__" << LR_dic.size() << std::endl;
  return (HR_dic.size() == LR_dic.size());
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
      retrieve a cv::Size patch with the keypoints as center.
      and then create dictionaries with similar patches but one is from the HR images
      and the other is from the same images but blurred.
    */
    for(unsigned int j = 0; j < keypoints.size(); ++j){

      getRectSubPix(data[i], Size(5,5), keypoints[j].pt, patch_HR);
      getRectSubPix(bl_data[i], Size(5,5), keypoints[j].pt, patch_HR_bl);

	  /*
	  Mat im_with_keypoints;
	  drawKeypoints( data[i], keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
	  imshow("keypoints", im_with_keypoints );
	  waitKey(0);
	  */

      //Lab values of the patch
      cvtColor(patch_HR, Lab, CV_BGR2Lab);
      cvtColor(patch_HR_bl, Lab_bl, CV_BGR2Lab);

      //X & Y Gradient of ROI(Sobel Derivative) on Lab values.
      Sobel( Lab, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
      Sobel( Lab, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );

      Sobel( Lab_bl, grad_bl_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
      Sobel( Lab_bl, grad_bl_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );

      //Construction of each information structure of the dictionarie
      HR = std::make_tuple(patch_HR, Lab, grad_x, grad_y);
      HR_bl = std::make_tuple(patch_HR_bl, Lab_bl, grad_bl_x, grad_bl_y);

      //Construction of the dictionaries
      HR_dic.push_back(HR);
      LR_dic.push_back(HR_bl);
    }
  }
}
