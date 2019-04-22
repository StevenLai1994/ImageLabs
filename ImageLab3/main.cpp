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
//1、利用均值模板平滑灰度图像。
//具体内容:利用 OpenCV 对图像像素进行操作,分别利用 3*3、 5*5 和 9*9
//尺寸的均值模板平滑灰度图像

Mat meanFilter(Mat src_img, int k_size) {
    Mat result_img;
    Mat kernel = Mat::ones(k_size, k_size, CV_32F);
    kernel /= (float)(k_size * k_size);
    filter2D(src_img, result_img, src_img.depth(), kernel);
    return result_img;
}

//2、利用高斯模板平滑灰度图像。
//具体内容:利用 OpenCV 对图像像素进行操作,分别利用 3*3、 5*5 和 9*9
//尺寸的高斯模板平滑灰度图像

Mat gaussianFilter(Mat src_img, float sigma, int k_size) {
    Mat kernel = Mat::zeros(k_size, k_size, CV_32F);
    int center = k_size/2;
    float sigma2 = sigma * sigma;
    float sum = 0;
    for(int i=0; i<kernel.rows; i++) {
        float py = pow(i - center, 2);
        for(int j=0; j<kernel.cols; j++) {
            float px = pow(j - center, 2);
            float up = exp(-(px + py) / (2 * sigma2));
            float val = up / (2 * CV_PI * sigma2);
            sum += val;
            kernel.at<float>(i, j) = val;
        }
    }

    //使模板值之和为1
    for(int i=0; i<kernel.rows; i++) {
        for(int j=0; j<kernel.cols; j++) {
            kernel.at<float>(i, j) /= sum;
        }
    }

    Mat result_img;
    filter2D(src_img, result_img, src_img.depth(), kernel);
    return result_img;
}
//3、利用 Laplacian、Robert、Sobel 模板锐化灰度图像。
//具体内容:利用 OpenCV 对图像像素进行操作,分别利用 Laplacian、 Robert、
//Sobel 模板锐化灰度图像
Mat laplacianFilter(Mat src_img) {
    Mat kernel = (Mat_<int>(3, 3) << 0, 1, 0,
                                     1, -4, 1,
                                     0, 1, 0 );

//    Mat kernel = (Mat_<int>(3, 3) << 1, 1, 1,
//                                     1, -8, 1,
//                                     1, 1, 1 );

    cout << kernel << endl;
    Mat result_img;
    filter2D(src_img, result_img, src_img.depth(), kernel,
             Point(-1, -1), BORDER_REPLICATE);
//    double minval, maxval;
//    cv::minMaxIdx(result_img, &minval, &maxval);
//    for(auto it=result_img.begin<float>(); it != result_img.end<float>(); it++) {
//        float val = (*it - minval) * 255 / (maxval - minval);
//        *it = val;
//    }
//    normalize(result_img, result_img, 0, 255, NORM_MINMAX);
//    Laplacian(src_img, result_img, src_img.depth());
//  normalize(result_img, result_img, 0, 255, NORM_MINMAX);
    cout << result_img << endl;
    return src_img - result_img;
}

Mat robertFilter(Mat src_img) {
    Mat kernel1 = (Mat_<int>(2, 2) << -1, 0,
                                     0, 1);
    Mat kernel2 = (Mat_<int>(2, 2) << 0, -1,
                                    1, 0);
    Mat result_img1, result_img2;
    filter2D(src_img, result_img1, src_img.depth(), kernel1,
             Point(-1, -1), 0, BORDER_REPLICATE);
    filter2D(src_img, result_img2, src_img.depth(), kernel2,
             Point(-1, -1), 0, BORDER_REPLICATE);

    return src_img - result_img1 - result_img2;
}

Mat sobelFilter(Mat src_img) {
    Mat kernel1 = (Mat_<int>(3, 3) << -1, -2, -1,
                                        0, 0, 0,
                                        1, 2, 1);
    Mat kernel2 = (Mat_<int>(3, 3) << -1, 0, -1,
                                        -2, 0, 2,
                                        -1, 0, 1);
    Mat result_img1, result_img2;
    filter2D(src_img, result_img1, src_img.depth(), kernel1,
             Point(-1, -1), 0, BORDER_REPLICATE);
    filter2D(src_img, result_img2, src_img.depth(), kernel2,
             Point(-1, -1), 0, BORDER_REPLICATE);

    return src_img - result_img1 - result_img2;
}
//4、利用增强灰度图像。
//具体内容:利用 OpenCV 对图像像素进行操作,设计高提升滤波算法增
//强图像
Mat highBoosting(Mat src_img, float sigma, int k_size) {
    //高斯平滑
    Mat result_img = gaussianFilter(src_img, sigma, k_size);
    draw(result_img, "1");
    //差值图 = 原图 - 平滑图
    result_img = src_img - result_img;
    draw(result_img, "2");
    //效果图 = 原图 + 差值图
    result_img += src_img;
    return result_img;
}
//5、利用均值模板平滑彩色图像。
//具体内容:利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作,利
//用 3*3、5*5 和 9*9 尺寸的均值模板平滑彩色图像
//6、利用高斯模板平滑彩色图像。
//具体内容:利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作,分
//别利用 3*3、5*5 和 9*9 尺寸的高斯模板平滑彩色图像
//7、利用 Laplacian、Robert、Sobel 模板锐化灰度图像。
//具体内容:利用 OpenCV 分别对图像像素的 RGB 三个通道进行操作,分
//别利用 Laplacian、Robert、Sobel 模板锐化彩色图像
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

//均值滤波灰度图
    draw(gray_img, "gray_img");
    waitKey();
    for(int i=3; i<=9; i+=2) {
        if(i == 7) continue;
        Mat result_img = meanFilter(gray_img, i);
        stringstream name;
        name << "mean filtering with kernel size " << i;
        draw(result_img, name.str());
    }
    waitKey();
    destroyAllWindows();


//高斯滤波灰度图
    draw(gray_img, "gray_img");
    waitKey();
    for(int i=3; i<=9; i+=2) {
        if(i == 7) continue;
        Mat result_img = gaussianFilter(gray_img, 1.5, i);
        stringstream name;
        name << "gaussian filtering with kernel size " << i;
        draw(result_img, name.str());
    }
    waitKey();
    destroyAllWindows();

//灰度图锐化
    draw(gray_img, "gray_img");
    waitKey();
    //拉普拉斯滤波
    Mat laplacian_img = laplacianFilter(gray_img);
    draw(laplacian_img, "Laplacian filtering");

    //罗伯特滤波
    Mat robert_img = robertFilter(gray_img);
    draw(robert_img, "Robert filtering");

    //索贝尔滤波
    Mat sobel_img = sobelFilter(gray_img);
    draw(sobel_img, "Sobel filtering");
    waitKey();
    destroyAllWindows();

//高提升滤波
    draw(gray_img, "gray_img");
    waitKey();
    Mat highB_img = highBoosting(gray_img, 1.5, 5);
    draw(highB_img, "high boosting filtering");
    waitKey();
    destroyAllWindows();

//利用均值模板平滑彩色图像
    draw(src_img, "src_img");
    waitKey();
    for(int i=3; i<=9; i+=2) {
        if(i == 7) continue;
        Mat result_img = meanFilter(src_img, i);
        stringstream name;
        name << "mean filtering with kernel size " << i;
        draw(result_img, name.str());
    }
    waitKey();
    destroyAllWindows();

//利用高斯模板平滑彩色图像
    draw(src_img, "src_img");
    waitKey();
    for(int i=3; i<=9; i+=2) {
        if(i == 7) continue;
        Mat result_img = gaussianFilter(src_img, 1.0, i);
        stringstream name;
        name << "gaussian filtering with kernel size " << i;
        draw(result_img, name.str());
    }
    waitKey();
    destroyAllWindows();

//利用 Laplacian、Robert、Sobel 模板锐化彩色图像。
    draw(src_img, "src_img");
    waitKey();

    //拉普拉斯滤波
    Mat RGBlaplacian_img = laplacianFilter(src_img);
    draw(RGBlaplacian_img, "Laplacian filtering");

    //罗伯特滤波
    Mat RGBrobert_img = robertFilter(src_img);
    draw(RGBrobert_img, "Robert filtering");

    //索贝尔滤波
    Mat RGBsobel_img = sobelFilter(src_img);
    draw(RGBsobel_img, "Sobel filtering");

    waitKey();
    destroyAllWindows();

    return 0;
}
