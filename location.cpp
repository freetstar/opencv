/*
 * =====================================================================================
 *
 *       Filename:  sobel.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04/09/2012 03:26:59 PM
 *       Revision:  none
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

using namespace cv;
using namespace std;

const int  EDGE_THREAD = 120;
const int  R           = 0.8;
const float  T         = 2.3;
const float THREAD_A   = 0.2;
const float THREAD_B   = 0.8;
const int BLOBNUM      = 300;
const float P =0.8;
const float Q =0.2;
const int S  = 10;

typedef Rect_<float> floatRect;

// Sobel operator implementation using inderect access
void Sobel(const  Mat &img, Mat &dst)
{
    //allocate if neccessary
    dst.create(img.size(),img.type());
    for (int i=1; i < img.rows-2; i++)
    {
        const uchar* previous = img.ptr<const uchar>(i-1);
        const uchar* current = img.ptr<const uchar>(i);
        const uchar* next = img.ptr<const uchar>(i+1);

        uchar* output = dst.ptr<uchar>(i);

        int x,y;
        for (int j = 1; j < img.cols-2; j++) 
        {
            //x=0;
            //x = int(previous[j-1]-previous[j+1]+2*current[j-1]-2*current[j+1]+next[j-1]-next[j+1]);
            y=0;
            y = int(previous[j-1]+2*previous[j]+previous[j+1]
                    -next[j-1]-2*next[j-1]-next[j+1]);
            output[j] = saturate_cast<uchar>(abs(y));
        }
    dst.row(0).setTo(Scalar(0));
    dst.row(dst.rows-1).setTo(Scalar(0));
    dst.col(0).setTo(Scalar(0));
    dst.col(dst.cols-1).setTo(Scalar(0));
   }
}

//使用阈值thread获得黑白图像
void Edge( Mat &img,Mat &dst, int thread)
{
    //allocate if neccessary
    dst.create(img.size(),img.type());

    int i=0;
    int j=0;

    for(i=1; i<img.rows;i++)
    {
        uchar* output = dst.ptr<uchar>(i);
        for(j=1; j<img.cols;j++)
        {
            if ( img.at<uchar>(i,j) < thread)
            {
                output[j] = 0;
            }
            else
                output[j] = 255;
        }
    }
}

//用水平窗口进行扫描,获取边缘密度
void EdgeIntensity( Mat &img,Mat &dst)
{
    //allocate if neccessary
    dst.create(img.size(),img.type());

    int i=0;
    int j=0;

    for(i=1; i<img.rows;i++)
    {
        uchar* output = dst.ptr<uchar>(i);

        for(j=1; j<img.cols;j++)
        {
          
          uchar sum=0;
          for(int k=-4;k<5;k++)
             sum += img.at<uchar>(i+k,j);
          if(sum > 245) output[j] = 255;
        }
    }
}

void Erease(Mat &img,Mat &dst)
{
    dst.create(img.size(),img.type());
    int i=0;
    int j=0;

    for(i=1;i<img.rows;i++)
    {
        uchar* output = dst.ptr<uchar>(i);
        for(j=1;j<img.cols;j++)
         {
             //先赋值称为黑色
             output[j]=0;
             for(int n=0;n<3;n++)
             {
                 //如果左右自身都不是黑色的点，则将目标点设为白色
                 if(img.at<uchar>(i,j+n-1)==255)
                 {output[j] = 255;break;}
             }
        }

    }
}

//形态学step1
void morphology1(Mat &img,Mat &dst)
{
    dst.create(img.size(),img.type());
    //闭运算
    dilate(img,dst,Mat());
    erode(dst,dst,Mat());

    //开运算
//    erode(dst,dst,Mat());
//    dilate(dst,dst,Mat());
}

//形态学第二步,找出必要的轮廓信息,然后找到连通域,进行下一步操作
void morphology2(const Mat &img,Mat &dst)
{
    vector< vector<Point> > contours;
    //find contours
    Mat imgclone = img.clone();
    dst.create(img.size(),img.type());
    findContours(imgclone,
            contours,//a vector of contours,which store the contours in a vector
            CV_RETR_LIST, //retrieve the external contours
            CV_CHAIN_APPROX_NONE); //all pixels of each contours

    vector<Rect> r(contours.size());

   for(int i = 0; i <contours.size();i++)
   {
      r[i]= boundingRect(Mat(contours[i]));
      int dilationWidth= min(r[i].height,r[i].width/2);
      int erosionWidth = r[i].width/2 ;
   }


   //画出轮廓信息
   drawContours(dst,contours,
           -1,//draw all contours
           Scalar(255), // in black
           2);
}


/* 寻找连通域，将连通域返回到blobs中，用floodfill来填充
 *  binary 是形态学第二部产生的结果图像
 *    blobs  连通域点的集合
 *  img    是原始图像的灰度图
 *  result  用来记录图像的文本框结果
 */
Mat FindBlobs(const Mat &binary, vector < vector<Point> > &blobs,const Mat &img,vector< floatRect > &result)
{
    blobs.clear();
    result.clear();
    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground
    Mat label_image = binary.clone();
    int label_count = 2; // starts at 2 because 0,1 are used already
    for(int y=0; y < binary.rows; y++) {
        for(int x=0; x < binary.cols; x++) {
            if((int)label_image.at<uchar>(y,x) != 255) {//不是白色，则跳过
                continue;
            }  
            //一个Rect类型的容器
            Rect rect;
            floodFill(label_image, Point(x,y), Scalar(label_count),&rect,1,3);
            //连通域blob是点的集合
            vector< Point> blob;
            for(int i=rect.y; i <=(rect.y+rect.height); i++) {
                for(int j=rect.x; j <=(rect.x+rect.width); j++) {
                    if((int)label_image.at<uchar>(i,j) != label_count ) {
                        continue;
                    }
                    blob.push_back(Point(j,i));
                }
            }
            if(blob.size()!=0)
            {
                blobs.push_back(blob);
            } 
            label_count++;
        }
    }

    //连通域的最小x和做大x
    //连通域的最小y和最大y
    int  ccxmin[BLOBNUM]; 
    int  ccxmax[BLOBNUM]; 
    int  ccymin[BLOBNUM]; 
    int  ccymax[BLOBNUM]; 
    //连通域的x方向，即宽度，连通域的y方向，即高度
    int  ccxwid[BLOBNUM]; 
    int  ccyhei[BLOBNUM]; 
    for(int i=0; i < blobs.size(); i++) {
        floatRect Recttemp;
        ccxmax[i] ={0};
        ccxmin[i] ={1500};
        ccymax[i] ={0};
        ccymin[i] ={1500};
        ccxwid[i] ={0};
        ccyhei[i] ={0};
        for(size_t j=0; j < blobs[i].size(); j++) {
            //x为图像所在的列，y为图像所在的行
            if (blobs[i][j].x>=ccxmax[i]) {ccxmax[i] = blobs[i][j].x;}
            if (blobs[i][j].x<=ccxmin[i]) {ccxmin[i] = blobs[i][j].x;}
            if (blobs[i][j].y>=ccymax[i]) {ccymax[i] = blobs[i][j].y;}
            if (blobs[i][j].y<=ccymin[i]) {ccymin[i] = blobs[i][j].y;}
        }
        //求出连通域的宽度和高度
        //这里可能出现max和min值相同的情况，暂且给他+1,不会有太大的影响
        ccxwid[i] = ccxmax[i]-ccxmin[i];
        ccyhei[i] = ccymax[i]-ccymin[i];
        if(ccxmax[i]==ccxmin[i])
        {
            ccxwid[i]=1;
        }
        if(ccymax[i]==ccymin[i])
        {
            ccyhei[i]=1;
        }
        //求出面积,x起始比例，y起始比例
        Recttemp.x = float(ccxmin[i])/binary.cols;
        Recttemp.y = float(ccymin[i])/binary.rows;
        Recttemp.width = float(ccxmax[i]-ccxmin[i])/binary.cols;
        Recttemp.height = float(ccymax[i]-ccymin[i])/binary.rows;
        result.push_back(Recttemp);
    }

    int whitedots[BLOBNUM] = {0};
    int blackdots[BLOBNUM] = {0};
    for(int m=0;m<blobs.size();m++)
    {
        float a;
        float b;
        //统计本连通域内的黑色点和白色点
        {
            for(int i=ccxmin[m]; i<=ccxmax[m];i++) {
                for(int j=ccymin[m]; j <= ccymax[m];j++) {
                    if((int)label_image.at<uchar>(j,i) == 255) {
                        whitedots[m] +=1;        
                    }
                    else
                        blackdots[m] +=1;
                }
            a = float(ccxwid[m])/float(ccyhei[m])+0.5;
            b = float(whitedots[m])/float(blackdots[m])+1.9;
         }
        }
        float cut;
        //条件一 用条件来判断颜色空间,返回cut
        {
            const int t=124;
            int num = 0;
            int chardots= 0;
            for(int i=ccxmin[m]; i<=ccxmax[m];i++) {
                for(int j=ccymin[m]; j <= ccymax[m];j++) {
                    if(int(img.at<uchar>(j,i)) >= t) {
                        chardots +=1;        
                    }
                    num ++;
                }
             }
            cut = float(chardots)/float(num);
        }
//        //条件二 用连通域分布来限制
//        {
//            if( b > 0.2 && b < 0.8)
//            {
//                cout<<"Gotcha"<<endl;
//            }
//            else if ( b >0.8 && ccxwid[m] > 0.5 )
//            else if (b)
//
//        }
            
        int peaknum = 0;
        float variance = 0.0;
        //条件三 用投影分析来做限制,返回peaknum
        {
            int num = 0; 
            int h[ccxwid[m]];
            for(int i=ccxmin[m]; i<=ccxmax[m];i++) 
            {
                for(int j=ccymin[m]; j<=ccymax[m];j++)
                {
                    if(int(img.at<uchar>(j,i)) == 255) {
                        h[num]++;          
                    }
                }
                num++;
            }
            //  此段函数用来获取整个数组的波峰，当然也可以获取整端函数的波谷
            int b[ccxwid[m]] ;
            b[0] = 1;
            int f = 1;
            for(int i=0;i<=num;i++)
            {
                if(h[i] > h[i-1]) {b[i] = 1;f =1;}
                else if (h[i] == h[i-1]) { b[i]= f;}
                else { b[i] = -1; f = -1;}
            }
            
            for(int i=0;i<num;i++)
            {
                if(b[i]+b[i+1]==0){ peaknum += 1;}
            }

            //获取曲线方差
            int sum= 0;
            for(int i=0;i<num;i++)
            {
                sum += h[i];
            }
            int mean = 0;
            mean = sum/ccxwid[m];
            for (int j = 0; j < num; j++) {
                variance += (h[j]-mean)*(h[j]-mean);
            }
            variance = variance / (ccxwid[m]*ccyhei[m]*ccyhei[m]);
        }


        //综合上述三个条件进行判断?:
        if(a > R && b > T)
        {
            if(cut > THREAD_A && cut <= THREAD_B )
            {
                if (peaknum < 5 ||variance < 0.05)
                {
                line(label_image,Point(ccxmin[m],ccymin[m]),Point(ccxmax[m],ccymin[m]),Scalar(255,0,0));
                line(label_image,Point(ccxmin[m],ccymax[m]),Point(ccxmax[m],ccymax[m]),Scalar(255,0,0));
                line(label_image,Point(ccxmin[m],ccymin[m]),Point(ccxmin[m],ccymax[m]),Scalar(255,0,0));
                line(label_image,Point(ccxmax[m],ccymin[m]),Point(ccxmax[m],ccymax[m]),Scalar(255,0,0));
                }
            }
        } 
    }            
 
    //返回图像
    return label_image;
    
}

//在一个name窗口中打开一个img图像
void showWindowImg(const char* name,Mat &img,int flag =0)
{
    namedWindow(name,flag);
    imshow(name,img);
}

Mat process(const Mat &src,vector< floatRect > &result)
{
    Mat out1,out2,out3,out4,out5;
    Sobel(src,out1);
    Edge(out1,out2,EDGE_THREAD);
    EdgeIntensity(out2,out3);
    morphology1(out3,out4);
    morphology2(out4,out5);
    showWindowImg("out5",out5);
    vector < vector <Point> > blobs;
    return FindBlobs(out5,blobs,src,result);
}

int intercourse(floatRect Rect1, floatRect Rect2)
{
    float minx = 0.0;
    float miny = 0.0;
    float maxx = 0.0;
    float maxy = 0.0;

    minx = Rect1.x >= Rect2.x ? Rect1.x : Rect2.x;
    miny = Rect1.y >= Rect2.y ? Rect1.y : Rect2.y;
    maxx = (Rect1.x + Rect1.width) <= (Rect2.x + Rect2.width)? (Rect1.x + Rect1.width):(Rect2.x + Rect2.width);
    maxy = (Rect1.y + Rect1.height) <= (Rect2.y + Rect2.height)? (Rect1.y + Rect1.height):(Rect2.y + Rect2.height);
    if(minx>maxx||miny>maxy)
    {
        return false;
    }
    else
    {
//        cout<<minx<<endl;
//        cout<<maxx<<endl;
//        cout<<maxy<<endl;
//        cout<<miny<<endl;
//        cout<<(maxx-minx)*(maxy-miny)<<endl;
        return 1000000*(maxx-minx)*(maxy-miny);
    }
}

int main(int argc, char* argv[])
{
    const Mat img = imread(argv[1],0);
    Mat imgclone = img.clone();
    //创建3个子图
    Mat div2img,div4img,div9img;
    //每个子图对应的输出结果
    Mat out,out2,out4,out9;
    resize(img,div2img,Size(),0.5,0.5);
    resize(img,div4img,Size(),0.25,0.25);
    resize(img,div9img,Size(),0.11,0.11);

        
    vector < floatRect > outRect;
    vector < floatRect > outRect2;
    out  = process(img,outRect);
    showWindowImg("out",out);
    out2 = process(div2img,outRect2);
    showWindowImg("out2",out2);

    for(int m=0;m<outRect.size();m++)
    {
        for (int n = 0; n < outRect2.size(); n++) {
            float size = float(intercourse(outRect[m],outRect2[n]))/1000000*out.cols*out.rows;
            cout<<size<<endl;

            //如果有交互，则运算
            if (size > 0)
            {
                    float tempmin = min(out.cols*out.rows*outRect[m].width*outRect[m].height,4*out2.cols*out2.rows*outRect2[n].width * outRect2[n].height); 
                    float tempmax = max(out.cols*out.rows*outRect[m].width*outRect[m].height,4*out2.cols*out2.rows*outRect2[n].width * outRect2[n].height);
                    if(size/tempmin > P  ) 
                    {    
                        cout<<"gotcha1"<<endl;
                        line(imgclone,Point(outRect[m].x*out.cols,outRect[m].y*out.rows),Point(outRect[m].x*out.cols+outRect[m].width*out.cols,outRect[m].y*out.rows),Scalar(255,255,0));
                        line(imgclone,Point(outRect[m].x*out.cols,outRect[m].y*out.rows),Point(outRect[m].x*out.cols,out.rows*outRect[m].y+outRect[m].height*out.rows),Scalar(255,255,0));
                        line(imgclone,Point(outRect[m].x*out.cols,out.rows*outRect[m].y+out.rows*outRect[m].height),Point(outRect[m].x*out.cols+outRect[m].width*out.cols,outRect[m].y*out.rows+outRect[m].height*out.rows),Scalar(255,255,0));
                        line(imgclone,Point(outRect[m].x*out.cols+out.cols*outRect[m].width,out.rows*outRect[m].y),Point(outRect[m].x*out.cols+outRect[m].width*out.cols,outRect[m].y*out.rows+outRect[m].height*out.rows),Scalar(255,255,0));
                    }
                    else if (size/tempmin > Q && tempmax/tempmin > S )
                    {    
                        cout<<"gotcha2"<<endl;
                        line(imgclone,Point(outRect[m].x*out.cols,outRect[m].y*out.rows),Point(outRect[m].x*out.cols+outRect[m].width*out.cols,outRect[m].y*out.rows),Scalar(255,255,0));
                        line(imgclone,Point(outRect[m].x*out.cols,outRect[m].y*out.rows),Point(outRect[m].x*out.cols,out.rows*outRect[m].y+outRect[m].height*out.rows),Scalar(255,255,0));
                        line(imgclone,Point(outRect[m].x*out.cols,out.rows*outRect[m].y+out.rows*outRect[m].height),Point(outRect[m].x*out.cols+outRect[m].width*out.cols,outRect[m].y*out.rows+outRect[m].height*out.rows),Scalar(255,255,0));
                        line(imgclone,Point(outRect[m].x*out.cols+out.cols*outRect[m].width,out.rows*outRect[m].y),Point(outRect[m].x*out.cols+outRect[m].width*out.cols,outRect[m].y*out.rows+outRect[m].height*out.rows),Scalar(255,255,0));
                    }
//                    cout<<"交合区域"<<size<<endl;
//                    cout<<"最小大小"<<min(1000000*outRect[m].width*outRect[m].height,1000000*outRect2[n].width * outRect2[n].height)<<endl;
            }
        }
    }

    showWindowImg("lined",imgclone);
//    out4  = process(div4img);
//    showWindowImg("out4",out4);
//    out9  = process(div9img);
//    showWindowImg("out9",out9);
    waitKey(0);
    return 0;
}
        
/*
 * 乎所有的文本定位算法都对字符大小很敏感, 为了能够找出大小不一的文本区域 , 我们采用金 字塔分解的方法 : 将图像分解为原分辨率的 1 四幅子图, 对每幅子图分别采用
 */

/*     Mat output = Mat::zeros(out5.size(), CV_8UC3);
 *     // Randomy color the blobs
 *     for(size_t i=0; i < blobs.size(); i++) {
 *         unsigned char r = 255 * (rand()/(1.0 + RAND_MAX));
 *         unsigned char g = 255 * (rand()/(1.0 + RAND_MAX));
 *         unsigned char b = 255 * (rand()/(1.0 + RAND_MAX));
 *  
 *         for(size_t j=0; j < blobs[i].size(); j++) {
 *             int x = blobs[i][j].x;
 *             int y = blobs[i][j].y;
 *  
 *             output.at<Vec3b>(y,x)[0] = b;
 *             output.at<Vec3b>(y,x)[1] = g;
 *             output.at<Vec3b>(y,x)[2] = r;
 *         }
 *     }
 *  
 *     showWindowImg("labelled", output);
 */
