#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>
#include <vector>
#include <tuple>

using namespace std;
using namespace cv;

typedef vector<std::tuple<Mat*, Mat*>> vTuple;
typedef vector<cv::Mat> vMat;

#define PATCH_SIZE 3

Scalar getMSSIM(Mat& i1, Mat& i2);
int patch_rank_on_ssim(vTuple &LR_dic, Mat &patchBicub);
int patch_rank_on_grad(vTuple &LR_dic, Mat &patchBicub);
int patch_rank_on_Lab(vTuple &LR_dic, Mat &patchBicub);
void superResolution(vTuple &HR_dic, vTuple &LR_dic);
void constructionDictionnaires(vTuple &HR_dic, vTuple &LR_dic);
void conv2(vMat &data, vMat &bl_data, int kernel_size);
void pickROI(vTuple &HR_dic, vTuple &LR_dic, vMat &data, vMat &bl_data);

int main(int argc, char** argv) {

	vTuple HR_dic, LR_dic;

	int choix = 0;

	while (true) {
		cout << "__ Veuillez choisir entre : \n __ la construction des dictionnaires (1) \n __ la super resolution d'une image (2)" << endl;
		cin >> choix;

		switch (choix) {
		case 1:
			constructionDictionnaires(HR_dic, LR_dic);
			cout << "SUCCESS" << endl;
			break;

		case 2:
			if (HR_dic.size() > 0 && LR_dic.size() > 0)
				superResolution(HR_dic, LR_dic);
			else {
				constructionDictionnaires(HR_dic, LR_dic);
				superResolution(HR_dic, LR_dic);
				return EXIT_SUCCESS;
			}
			break;

		default:
			break;
		}
	}
}

Scalar getMSSIM(Mat& i1, Mat& i2) {
	const double C1 = 6.5025, C2 = 58.5225;
	/***************************** INITS **********************************/
	int d = CV_32F;

	Mat I1, I2;
	i1.convertTo(I1, d);           // cannot calculate on one byte large values
	i2.convertTo(I2, d);

	Mat I2_2 = I2.mul(I2);        // I2^2
	Mat I1_2 = I1.mul(I1);        // I1^2
	Mat I1_I2 = I1.mul(I2);        // I1 * I2

	/*************************** END INITS **********************************/

	Mat mu1, mu2;   // PRELIMINARY COMPUTING
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);

	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);

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

	Scalar mssim = mean(ssim_map); // mssim = average of ssim map
	return mssim;
}

void superResolution(vTuple &HR_dic, vTuple &LR_dic) {

	Mat im_source = imread("../data/res/source.png");
	Mat le_patch;

	Mat im_dest;
	Mat patch;
	int index = -1;

	resize(im_source, im_dest, Size(), 2, 2, INTER_CUBIC);
	Mat im_final(Size(im_dest.cols, im_dest.rows), im_dest.type());


	#pragma omp parallel for
		for (int i = 0; i < im_dest.rows; i += PATCH_SIZE) {
			for (int j = 0; j < im_dest.cols; j += PATCH_SIZE) {
				if (j + PATCH_SIZE > im_dest.cols)break;
				getRectSubPix(im_dest, cv::Size(PATCH_SIZE, PATCH_SIZE), Point2f(i, j), patch);
				index = patch_rank_on_Lab(LR_dic, patch);
				//std::cout << index << std::endl;
				le_patch = *get<0>(HR_dic[index]);
				le_patch.copyTo(im_final(Rect(j, i, le_patch.cols, le_patch.rows)));
			}
			if (i + PATCH_SIZE > im_dest.rows)break;
		}

		Mat im;
		cvtColor(im_final, im, CV_Lab2BGR);
		imwrite("../data/res/res.png", im);
		imwrite("../data/res/src_bicub.png", im_dest);
}

inline
int patch_rank_on_ssim(vTuple &LR_dic, Mat &patchBicub){
	/*
	* score1 : SSIM sur les patchs*/

	double score = std::numeric_limits<double>::max();

	Scalar ssim_score, prec;
	int index = -1;
	double mean_actu, mean_prec = 0;

	for (unsigned int i = 0; i < LR_dic.size(); i++){
 		ssim_score = getMSSIM(*get<0>(LR_dic[i]), patchBicub);
 		mean_actu = ( ssim_score[0] + ssim_score[1] + ssim_score[2] )/3;
 		if(mean_actu > mean_prec) { index = i; mean_prec = mean_actu; }
	}

	return index;
}

int patch_rank_on_grad(vTuple &LR_dic, Mat &patchBicub) {
	/*
	 * score2 : distance euclidienne entre les couples de gradient*/

	int index = -1;
	double score = std::numeric_limits<double>::max(), prec = std::numeric_limits<double>::max();
	// Mat patch_gray;
	Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad;

	// cvtColor(patchBicub, patch_gray, CV_BGR2GRAY);

	//Gradient of bicubic patch
	Sobel(patchBicub, grad_x, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	Sobel(patchBicub, grad_y, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	for (unsigned int i = 0; i < LR_dic.size(); i++) {
		score = norm(*get<1>(LR_dic[i]), grad, NORM_L2);
		if (score < prec) { prec = score; index = i; }
	}
	return index;
}

int patch_rank_on_Lab(vTuple &LR_dic, Mat &patchBicub) {
	/*
	 * score3 : Score sur la valeur L des patchs Lab*/

	double score = std::numeric_limits<double>::max(), L_score = 0;
	int index = -1;
	int histSize = 256;

	//L varies from 0 to 255
	float L_ranges[] = { 0, 256 };
	const float* ranges = { L_ranges };

	//Histograms
	MatND hist_bicub;
	MatND hist_dic;

	Mat patch_Lab;
	cvtColor(patchBicub, patch_Lab, CV_BGR2Lab);

	vector<Mat> Lab_planes;
	vector<Mat> Lab_planes_patch;
	split(patch_Lab, Lab_planes);

	//Calculate the histograms for the Lab images
	calcHist(&Lab_planes[0], 1, 0, Mat(), hist_bicub, 1, &histSize, &ranges, true, false);
	normalize(hist_bicub, hist_bicub, 0, 1, NORM_MINMAX, -1, Mat());

	for (unsigned int i = 0; i < LR_dic.size(); i++) {
		split(*get<0>(LR_dic[i]), Lab_planes_patch);

		calcHist(&Lab_planes_patch[0], 1, 0, Mat(), hist_dic, 1, &histSize, &ranges, true, false);
		normalize(hist_dic, hist_dic, 0, 1, NORM_MINMAX, -1, Mat());

		//1 = Correlation		//2 = Chi-Square		//3 = Intersection		//4 = Bhattacharyya
		L_score = compareHist(hist_dic, hist_bicub, 4);
		if (L_score < score) { index = i; score = L_score; }
	}
	return index;
}

void constructionDictionnaires(vTuple &HR_dic, vTuple &LR_dic) {

	cv::String path("../data/training/*.bmp"); //select only bmp.

	vector<cv::String> fn;
	vMat data, bl_data; //Blurred HR Images.

	glob(path, fn, false); // recurse.

	for (unsigned int k = 0; k < fn.size(); ++k) {
		Mat im = imread(fn[k]);
		if (!im.data) cout << "Could not open or find an image" << endl;  // Check for invalid input.
		data.push_back(im);
	}

	//Blurring the HR images.
	conv2(data, bl_data, 3);
	if (data.size() != bl_data.size()) cerr << "Error during convolution : number of images is irrelevant.";

	pickROI(HR_dic, LR_dic, data, bl_data);

	assert(HR_dic.size() == LR_dic.size());
}

void conv2(vMat &data, vMat &bl_data, int kernel_size) {
	Mat dst, kernel;
	kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);

	/// Apply filter
	for (unsigned int i = 0; i < data.size(); ++i) {
		filter2D(data[i], dst, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
		bl_data.push_back(dst);
	}
}

void pickROI(vTuple &HR_dic, vTuple &LR_dic, vMat &data, vMat &bl_data) {

	Mat patch_HR, patch_HR_bl;
	Mat grad_x, grad_y, grad_bl_x, grad_bl_y;
	Mat abs_grad_x, abs_grad_y, abs_grad_bl_x, abs_grad_bl_y;

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

	for (unsigned int i = 0; i < data.size(); ++i) {
		// Detect blobs of important area in not blurred picture i.
		vector<KeyPoint> keypoints;
		detector->detect(data[i], keypoints);

		/*
		  retrieve a cv::Size patch with the keypoints as center.
		  and then create dictionaries with similar patches but one is from the HR images
		  and the other is from the same images but blurred.
		*/
		for (unsigned int j = 0; j < keypoints.size(); ++j) {

			getRectSubPix(data[i], Size(PATCH_SIZE, PATCH_SIZE), keypoints[j].pt, patch_HR);
			getRectSubPix(bl_data[i], Size(PATCH_SIZE, PATCH_SIZE), keypoints[j].pt, patch_HR_bl);

			//imshow("patch", patch_HR );
			//waitKey(1);

			//Lab values of the patch
			Mat *Lab = new Mat();
			Mat *Lab_bl = new Mat();
			cvtColor(patch_HR, *Lab, CV_BGR2Lab);
			cvtColor(patch_HR_bl, *Lab_bl, CV_BGR2Lab);

			//X & Y Gradient of ROI(Sobel Derivative) on Lab values.
			// Mat patch_HR_gray;
			// cvtColor(patch_HR, patch_HR_gray, CV_BGR2GRAY);

			Sobel(patch_HR, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
			Sobel(patch_HR, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
			convertScaleAbs(grad_x, abs_grad_x);
			convertScaleAbs(grad_y, abs_grad_y);

			//X & Y Gradient of blurred ROI(Sobel Derivative) on Lab values.
			// Mat patch_HR_gray_bl;
			// cvtColor(patch_HR_bl, patch_HR_gray_bl, CV_BGR2GRAY);

			Sobel(patch_HR_bl, grad_bl_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
			Sobel(patch_HR_bl, grad_bl_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
			convertScaleAbs(grad_bl_x, abs_grad_bl_x);
			convertScaleAbs(grad_bl_y, abs_grad_bl_y);

			Mat *grad = new Mat();
			Mat *grad_bl = new Mat();
			addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, *grad);
			addWeighted(abs_grad_bl_x, 0.5, abs_grad_bl_y, 0.5, 0, *grad_bl);

			std::tuple<Mat*, Mat*> hr_pair = make_tuple(Lab, grad);
			std::tuple<Mat*, Mat*> lr_pair = make_tuple(Lab_bl, grad_bl);

			//Construction of the dictionaries
			LR_dic.push_back(lr_pair);
			HR_dic.push_back(hr_pair);

			// namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
			// imshow("Display window", Lab );
			// waitKey(1);

		}
	}
}
