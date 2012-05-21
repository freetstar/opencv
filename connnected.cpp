/*
 * =====================================================================================
 *
 *       Filename:  connected.cpp
 *
 *    Description:  使用opencv的floodfill方法来获取连通域，为每个连通域设置一个label，
 *                  main函数中根据label进行颜色设置。
 *                  很蛋疼的是，http://nghiaho.com/?p=1102不能正确得到结果（二值图像明明
 *                  是0/255,作者却说0/1） 。在其基础上进进行了修改
 *
 *        Version:  1.0
 *        Created:  04/09/2012 03:26:59 PM
 *       Revision:  opencv2.3.1
 *       Compiler:  gcc
 *
 *         Author:  freetstar (http://www.freetstar.com), lgxwqq@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

void FindBlobs(const cv::Mat &binary, vector < vector<cv::Point>  > &blobs)
{
    blobs.clear();
 
    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground
 
    cv::Mat label_image = binary.clone();
   
    int label_count = 2; // starts at 2 because 0,1 are used already
 
    for(int y=0; y < binary.rows; y++) {
        for(int x=0; x < binary.cols; x++) {
            if((int)label_image.at<uchar>(y,x) != 255) {
                continue;
            }
 
            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), cv::Scalar(label_count), 
                          &rect, cv::Scalar(0), cv::Scalar(0), 4);
 
            vector< cv::Point>   blob;
 
            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if((int)label_image.at<uchar>(i,j) != label_count) {
                        continue;
                    }
 
                    blob.push_back(cv::Point(j,i));
                }
            }
 
            blobs.push_back(blob);
 
            label_count++;
        }
    }
}


int main(int argc, char **argv)
{
    cv::Mat img = cv::imread("blob.png", 0); // force greyscale
 
    if(!img.data) {
        cout << "File not found" << endl;
        return -1;
    }
 
    cv::namedWindow("binary");
    cv::namedWindow("labelled");
 
    cv::Mat output = cv::Mat::zeros(img.size(), CV_8UC3);
 
    cv::Mat binary = img.clone();
    vector < vector <cv::Point> > blobs;
 
    //cv::threshold(img, binary, 0.0, 1.0, cv::THRESH_BINARY);

 
    FindBlobs(binary, blobs);
 
    // Randomy color the blobs
    for(size_t i=0; i < blobs.size(); i++) {
        unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
        unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));
 
        for(size_t j=0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;
 
            output.at<cv::Vec3b>(y,x)[0] = b;
            output.at<cv::Vec3b>(y,x)[1] = g;
            output.at<cv::Vec3b>(y,x)[2] = r;
        }
    }
 
    cv::imshow("binary", img);
    cv::imshow("labelled", output);
    cv::waitKey(0);
 
    return 0;
}
