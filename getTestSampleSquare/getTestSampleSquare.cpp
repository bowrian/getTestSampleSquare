#include <Windows.h>
#include<algorithm>
#pragma comment( lib, "User32.lib")
#include "highgui/highgui.hpp"    
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"   
#include <iostream>  
using namespace cv;
using namespace std;
Mat ostu(Mat image02);//output is image02_binary
bool surfMatch(Mat image01,Mat image01_binary,Mat image02,Mat &image1t2_binary);//imgage01 is template ,image02 is a arbitary image to be tested.
Mat xor(Mat image02_binary,Mat image1t2_binary);//output is image02_result
void contourExtract(Mat image02_result,Mat image02);//the function is to detect the proper contour and save the roi which is in the original image02.
int main(int argc, char *argv[])
{
    Mat image01 = imread("images/image01.jpg"); 
	Mat image01_binary=imread("images/image01_binary.jpg",0); 
    //Mat image02 = imread("images/h2.34p2a45.jpg");
	Mat image02 = imread("images/image02.jpg");
	Mat image02_binary(image02.size(),CV_8UC1),image1t2_binary(image02.size(),CV_8UC1),image02_result;
	Mat MORPH_DILATE2;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
    //高级形态学处理，调用这个函数就可以了，具体要选择哪种操作，就修改第三个参数就可以了。
	morphologyEx(image02, MORPH_DILATE2,MORPH_DILATE, element);
	image02_binary=ostu(MORPH_DILATE2);
	//imwrite("images/image02_binary.jpg",image02_binary);
	imshow("image01",image01);
	imshow("image01_binary",image01_binary);
	imshow("image02",image02);
	imshow("image02_binary",image02_binary);
	if(waitKey(0)>0);
	if(surfMatch(image01,image01_binary,image02, image1t2_binary)==true)
	{
		//imshow("image1t2_binary",image1t2_binary);
		//if(waitKey(0)>0);
	}
	//imwrite("images/image1t2_binary.jpg",image1t2_binary);
	image02_result=xor(image02_binary,image1t2_binary);
	imshow("image02_result",image02_result);
	if(cv::waitKey(0)>0);
	contourExtract(image02_result,image02);
    return 0;
}
Mat ostu(Mat image02)//output is image02_binary
{
	IplImage * src= &IplImage(image02);
	int height = src->height;
    int width = src->width;
    long size = height * width;

    //histogram    
    float histogram[256] = { 0 };
    for (int m = 0; m < height; m++)
    {
        unsigned char* p = (unsigned char*)src->imageData + src->widthStep * m;
        for (int n = 0; n < width; n++)
        {
            histogram[int(*p++)]++;
        }
    }

    int threshold;
    long sum0 = 0, sum1 = 0; //存储前景的灰度总和和背景灰度总和  
    long cnt0 = 0, cnt1 = 0; //前景的总个数和背景的总个数  
    double w0 = 0, w1 = 0; //前景和背景所占整幅图像的比例  
    double u0 = 0, u1 = 0;  //前景和背景的平均灰度  
    double variance = 0; //最大类间方差  
    int i, j;
    double u = 0;
    double maxVariance = 0;
    for (i = 1; i < 256; i++) //一次遍历每个像素  
    {
        sum0 = 0;
        sum1 = 0;
        cnt0 = 0;
        cnt1 = 0;
        w0 = 0;
        w1 = 0;
        for (j = 0; j < i; j++)
        {
            cnt0 += histogram[j];
            sum0 += j * histogram[j];
        }

        u0 = (double)sum0 / cnt0;
        w0 = (double)cnt0 / size;

        for (j = i; j <= 255; j++)
        {
            cnt1 += histogram[j];
            sum1 += j * histogram[j];
        }

        u1 = (double)sum1 / cnt1;
        w1 = 1 - w0; // (double)cnt1 / size;  

        u = u0 * w0 + u1 * w1; //图像的平均灰度  
        printf("u = %f\n", u);
        //variance =  w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2);  
        variance = w0 * w1 *  (u0 - u1) * (u0 - u1);
        if (variance > maxVariance)
        {
            maxVariance = variance;
            threshold = i;
        }
    }
	Mat image02_binary(image02.size(),CV_8UC1);
	cvtColor(image02, image02_binary, COLOR_BGR2GRAY);  
	cv::threshold(image02_binary, image02_binary, threshold, 255,THRESH_BINARY );
	return image02_binary;
}
bool surfMatch(Mat image01,Mat image01_binary,Mat image02,Mat &image1t2_binary)//imgage01 is template ,image02 is a random image to be tested.
{
    //灰度图转换  
    Mat image1, image2;
    cvtColor(image01, image1, CV_RGB2GRAY);
    cvtColor(image02, image2, CV_RGB2GRAY);
    //提取特征点    
    SurfFeatureDetector Detector(2000);  
    vector<KeyPoint> keyPoint1, keyPoint2;
    Detector.detect(image1, keyPoint1);
    Detector.detect(image2, keyPoint2);

    //特征点描述，为下边的特征点匹配做准备    
    SurfDescriptorExtractor Descriptor;
    Mat imageDesc1, imageDesc2;
    Descriptor.compute(image1, keyPoint1, imageDesc1);
    Descriptor.compute(image2, keyPoint2, imageDesc2);

    FlannBasedMatcher matcher;
    vector<vector<DMatch> > matchePoints;
    vector<DMatch> GoodMatchePoints;

    vector<Mat> train_desc(1, imageDesc1);
    matcher.add(train_desc);
    matcher.train();

    matcher.knnMatch(imageDesc2, matchePoints, 2);
    cout << "total match points: " << matchePoints.size() << endl;

    // Lowe's algorithm,获取优秀匹配点
    for (int i = 0; i < matchePoints.size(); i++)
    {
        if (matchePoints[i][0].distance < 0.4* matchePoints[i][1].distance)
        {
            GoodMatchePoints.push_back(matchePoints[i][0]);
        }
    }
	cout<<"good match points:"<<GoodMatchePoints.size()<<endl;
	if(GoodMatchePoints.size()>=30)
	{
		Mat first_match;
		drawMatches(image02, keyPoint2, image01, keyPoint1, GoodMatchePoints, first_match);
		imshow("first_match ", first_match);
		if(waitKey(0)>0);
		vector<Point2f> imagePoints1, imagePoints2;
		for (int i = 0; i<GoodMatchePoints.size(); i++)
		{
			imagePoints2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx].pt);
			imagePoints1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx].pt);
		}
		//获取图像1到图像2的投影映射矩阵 尺寸为3*3  
		Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
		////也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差  
		//Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
		cout << "变换矩阵为：\n" << homo << endl << endl; //输出映射矩阵      
   
		//图像配准  
		warpPerspective(image01_binary, image1t2_binary, homo, Size(image02.cols, image02.rows));
		//warpPerspective(image01_binary, image1t2_binary, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), image02.rows));
		/*double adjustValue=image1.cols;  
		Mat adjustMat=(Mat_<double>(3,3)<<1.0,0,35,0,1.0,65,0,0,1.0);  
		warpPerspective(image01_binary, image1t2_binary, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));*/
		return true;
	}
	else
	{
		return false;
	}
}
Mat xor(Mat image02_binary,Mat image1t2_binary)
{
	Mat img1;
	img1=image1t2_binary;
	//cvtColor(image1t2_binary, img1, COLOR_BGR2GRAY);
    Mat img2;
	img2=image02_binary;
	//cvtColor(image02_binary, img2, COLOR_BGR2GRAY);
	int width=img1.cols;
	int height=img1.rows;
	int y,x;
	uchar *p1,*p2;
	for(y=0;y<height;y++)
	{
		p1=img1.ptr<uchar>(y);
		p2=img2.ptr<uchar>(y);
		for(x=0;x<width;x++)
		{
			if(p1[x]>=128)
			{
				p2[x]=0;
			}
		}
	}
	return img2;
}
void contourExtract(Mat image02_result,Mat image02)
{
	vector<vector<Point> > contours;   //轮廓数组
	vector<vector<Point> >::iterator itr;  //轮廓迭代器
	vector<vector<Point> > con;    //当前轮廓
	double minArea = 200,maxArea = 10000;
	double minLength=100,maxLength=2000;
	double minRate=2.0,maxRate=10.0;
	/*double minArea = 20,maxArea = 100000;
	double minLength=20,maxLength=20000;
	double minRate=1.0,maxRate=100.0;*/
	Mat image02_result_tmp(image02_result.size(),CV_8UC1);
	for (int y2 = 0; y2<image02_result.rows; ++y2)
	{
		for (int x2 = 0; x2<image02_result.cols; ++x2)
		{
			image02_result_tmp.at<uchar>(y2,x2)=image02_result.at<uchar>(y2,x2);
		}
	}
	Mat dst;
    vector<Vec4i> hierarchy;  
    findContours(image02_result_tmp,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_NONE,Point()); 
    int toatal = 0;  
    int index = 0;  
	Mat src=image02;
	std::cout<<"contours size:"<<contours.size()<<endl;
    for(int i=0;i<contours.size();i++)  
    {  
        int  length = contours[i].size();  
        if(length > minLength&&length<maxLength&&contourArea(contours[i])>minArea&&contourArea(contours[i])<maxArea)  
        {  
            index = i;
            //绘制轮廓的最小外结矩形  
            RotatedRect rect=minAreaRect(contours[i]);  
            double x0,y0,alpha,width,height;  
            x0 = rect.center.x;                    //  中心位置  
            y0 = rect.center.y;  
			width=rect.size.width;
			height=rect.size.height;
            alpha = -rect.angle*CV_PI/180;         //  角度转换成弧度  
			Point2f vertices[4];
			rect.points(vertices);
			Point2i  centerPoint = rect.center;
			Point2i ltPoint;
			ltPoint = vertices[1];
			double rate;
			if(width>height)
			{
				rate=width/height;
			}
			else
			{
				rate=height/width;
			}
			
			if(rate>minRate&&rate<maxRate)
			{
				toatal++;
				Rect boundRect=boundingRect(Mat(contours[i]));
				int len;
				if(boundRect.width>boundRect.height)
				{
					len=boundRect.width;
				}
				else
				{
					len=boundRect.height;
				}
				int center_x=(boundRect.tl().x+boundRect.br().x)/2;
				int center_y=(boundRect.tl().y+boundRect.br().y)/2;
				Mat ROI1=Mat(Size(len,len),CV_8UC1);
				int tl_x=std::min(std::max(center_x-len/2,0),image02_result.cols-len);
				int tl_y=std::min(std::max(center_y-len/2,0),image02_result.rows-len);
				Mat image02_clip_tmp(image02_result.size(),CV_8UC3);
				for (int y2 = 0; y2<image02_clip_tmp.rows; ++y2)
				{
					for (int x2 = 0; x2<image02_clip_tmp.cols; ++x2)
					{
						image02_clip_tmp.at<Vec3b>(y2,x2)=image02.at<Vec3b>(y2,x2);
					}
				}
				ROI1 = image02_clip_tmp(Rect(tl_x,tl_y,len,len));
				for(int y=0;y<len;y++)
				{
					for(int x=0;x<len;x++)
					{
						if(pointPolygonTest( contours[i], Point2i(x+tl_x,y+tl_y), true )>=0)
						{
							//ROI1.at<Vec3b>(y, x) = image02_result.at<Vec3b>(y+tl_y, x+tl_x);
						}
						else
						{
							ROI1.at<Vec3b>(y, x) =0;
						}
					}
				}
				/*cv::imshow("ROI",ROI1);
				if(cv::waitKey(0)>0);*/
				std::stringstream StrStm;
				StrStm.clear();
				StrStm << "images\\testImage\\ROI_"<<toatal << ".jpg" << endl;
				string filename;
				filename.clear();
				StrStm >> filename;
				cv::imwrite(filename, ROI1);
				
			}
        }  
    }  
	drawContours( image02_result, contours, 1, Scalar(200), 1, 8, vector<Vec4i>(), 0, Point() );
	cv::imshow("src",src);
	cv::imshow("imag02e_result",image02_result_tmp);
	if(cv::waitKey(0)>0);
}