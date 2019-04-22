#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

string imageDir = "/home/jj/Pictures/";

string imageName = "lena.png";

void draw(Mat img, string name) {
    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name,img);
}

Mat getHist(Mat img) {
    int channels[] = {0};
    int histSize[] = {256};
    float range[] = {0, 256};
    const float* ranges[] = {range};
    Mat hist;

    cv::calcHist(&img, 1, channels, Mat(), hist, 1, histSize, ranges);

    //maxValue是最大的频度，四舍五入得到 rows 是直方图的高度
    double maxValue;
    cv::minMaxLoc(hist, 0, &maxValue);
    int rows=cvRound(maxValue);

    //用来画直方图，底黑柱白
    Mat histImage = Mat::zeros(rows,256,CV_8UC1);

    //遍历每个像素值，画出相应频度的高度
    for(int i=0; i<256; i++) {
        int temp = (int)(hist.at<float>(i, 0));
         if(temp)
            histImage.col(i).rowRange(Range(rows-temp, rows)) = 255;
    }
    resize(histImage, histImage, img.size());
    return histImage;
}

Mat mySingleEqualizeHist(Mat single_channel_img) {
    int channels[] = {0};
    int histSize[] = {256};
    float range[] = {0, 256};
    const float* ranges[] = {range};
    Mat hist;
    cv::calcHist(&single_channel_img, 1, channels, Mat(), hist, 1, histSize, ranges);
    Mat table = Mat(1, 256, CV_8U);
    uchar* r = table.data;
    float dim = single_channel_img.rows * single_channel_img.cols;
    float Si = 0.0;
    for(int i=0; i<256; i++) {
        Si += (hist.at<float>(i, 0)) / dim;
        r[i] = (uchar)(255 * Si);
    }
    Mat result_img;
    LUT(single_channel_img, table, result_img);
    return result_img;
}

Mat myRGBEqualizeHist(Mat RGB_img) {
    //拆分通道
    vector<Mat> channels;
    split(RGB_img, channels);
    vector<Mat> new_channels;
    new_channels.push_back(mySingleEqualizeHist(channels[0]));
    new_channels.push_back(mySingleEqualizeHist(channels[1]));
    new_channels.push_back(mySingleEqualizeHist(channels[2]));
    Mat result_img;

    //合并通道
    merge(new_channels, result_img);
    return result_img;
}

int main()
{
    cout<<"file name:"<<endl;

    //cin>>imageName;
    string filePath=imageDir + imageName;
    cout<<"The input is "<<filePath<<endl;

    Mat src_img=imread(filePath), gray_img;
    if(src_img.empty())
    {
        cout<<"image is empty!"<<endl;
    }
    cvtColor(src_img, gray_img, COLOR_RGB2GRAY);

    draw(src_img, "src_img");
    draw(gray_img, "gray_img");
    waitKey();
    //显示直方图
    Mat hist_img = getHist(gray_img);
    draw(hist_img, "hist_img");
    waitKey();
    
//    //直方图均衡化
    Mat api_avg_img,api_avg_hist_img;
    equalizeHist(gray_img, api_avg_img);
    api_avg_hist_img = getHist(api_avg_img);
    draw(api_avg_img, "api_avg_img");
    draw(api_avg_hist_img, "api_avg_hist_img");
    waitKey();

//    //自定义灰度图直方图均衡化
    Mat my_avg_img,my_avg_hist_img;
    my_avg_img = mySingleEqualizeHist(gray_img);
    my_avg_hist_img = getHist(my_avg_img);
    draw(my_avg_img, "my_avg_img");
    draw(my_avg_hist_img, "my_avg_hist_img");
    waitKey();
    //自定义RGB图直方图均衡化
    Mat my_RGB_avg_img,my_RGB_avg_hist_img;
    my_RGB_avg_img = myRGBEqualizeHist(src_img);
    my_RGB_avg_hist_img = getHist(my_RGB_avg_img);
    draw(my_RGB_avg_img, "my_RGB_avg_img");
    draw(my_RGB_avg_hist_img, "my_RGB_avg_hist_img");

    waitKey();
    destroyAllWindows();
    return 0;
}
