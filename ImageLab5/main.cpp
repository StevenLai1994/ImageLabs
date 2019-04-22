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

//1、灰度图像的DFT和IDFT。具体内容：利用OpenCV提供的cvDFT函数对图像进行DFT和IDFT变换
void swapMat(Mat& src_img) {
    src_img = src_img(Rect(0, 0, src_img.cols&-2, src_img.rows&-2));
    int cx = src_img.cols/2, cy = src_img.rows/2;
    Mat q0(src_img, Rect(0, 0, cx, cy));       //左上角图像划定ROI区域
    Mat q1(src_img, Rect(cx, 0, cx, cy));      //右上角图像
    Mat q2(src_img, Rect(0, cy, cx, cy));      //左下角图像
    Mat q3(src_img, Rect(cx, cy, cx, cy));     //右下角图像
    Mat tmp;

    //变换左上角和右下角
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    //变换右上角和左下角象限
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
Mat myDFT(Mat src_img) {
    Mat temp;
    //寻找FFT计算最快的尺寸
    int m = getOptimalDFTSize(src_img.rows);
    int n = getOptimalDFTSize(src_img.cols);
    Mat padded;
    //边界补齐
    copyMakeBorder(src_img, padded, 0, m-src_img.rows,
                   0, n-src_img.cols, BORDER_CONSTANT, Scalar::all(0));
    //图像补一个尺寸相同的虚部通道
    Mat fill_imgs[] {Mat_<float>(src_img), Mat::zeros(src_img.size(), CV_32F)};
    merge(fill_imgs, 2, temp);
    Mat result_img;
    dft(temp, result_img);
    return result_img;
}
//获取傅里叶变换的幅度谱
Mat getAmplitude(Mat dft_img) {
    Mat channels[2];
    split(dft_img, channels);
    magnitude(channels[0], channels[1], channels[0]);
    channels[0] += Scalar::all(1);
    log(channels[0], channels[0]);
    normalize(channels[0], channels[0], 0, 1, CV_MINMAX);
    return channels[0];
}
//傅里叶逆变换
Mat myIDFT(Mat dft_img) {
    Mat result_img;
    idft(dft_img, result_img, DFT_REAL_OUTPUT);
    normalize(result_img, result_img, 0, 1, CV_MINMAX);
    //cout << result_img(Rect(0, 0, 10, 10)) << endl;
    return result_img;
}

//2、利用理想高通和低通滤波器对灰度图像进行频域滤波
//具体内容：
//利用cvDFT函数实现DFT，在频域上利用理想高通和低通滤波器进行滤波，
//并把滤波过后的图像显示在屏幕上（观察振铃现象），要求截止频率可输入。
Mat idealFilter(Mat dft_img, string flag, float D0) {
    Mat temp_img = dft_img.clone();
    Mat result_img;
    swapMat(temp_img);
    int cx = temp_img.cols / 2, cy = temp_img.rows / 2;
    for(int y=0; y<temp_img.rows; y++) {
        for(int x=0; x<temp_img.cols; x++) {
            float d = sqrt(pow(x - cx, 2) + pow(y - cy, 2));
            if(flag == "low") {
                //低通
                if(d >= D0) {
                    temp_img.at<Vec2f>(x, y) = 0;
                }
            }
            else {
                //高通
                if(d <= D0) {
                    temp_img.at<Vec2f>(x, y) = 0;
                }
            }
        }
    }
    draw(getAmplitude(temp_img), "ideal "+flag+" Amplitude");
    //cout << flag+ "_img: " << getAmplitude(temp_img)(Rect(0, 0, 10, 10)) << endl;
    swapMat(temp_img);
    result_img =  myIDFT(temp_img);
    return result_img;
}

//3、利用布特沃斯高通和低通滤波器对灰度图像进行频域滤波。
//具体内容：利用cvDFT函数实现DFT，在频域上进行利用布特沃斯高通和低通滤波器进行滤波，
//并把滤波过后的图像显示在屏幕上（观察振铃现象），要求截止频率和n可输入。
Mat ButterworthFilter(Mat dft_img, float D0, float n, string flag) {
    Mat temp_img = dft_img.clone();
    Mat result_img;
    swapMat(temp_img);
    int cx = temp_img.cols / 2, cy = temp_img.rows / 2;
    for(int y=0; y<temp_img.rows; y++) {
        for(int x=0; x<temp_img.cols; x++) {
            float d = sqrt(pow(x - cx, 2) + pow(y - cy, 2));
            if(flag == "low") {
                //低通
                temp_img.at<Vec2f>(x, y) *= 1.0 / (1 + pow((d/D0), 2*n));
            }
            else {
                //高通
                temp_img.at<Vec2f>(x, y) *= (1 - 1.0 / (1 + pow((d/D0), 2*n)));
            }
        }
    }
    draw(getAmplitude(temp_img), flag + "ButterworthAmplitude");
    swapMat(temp_img);
    result_img =  myIDFT(temp_img);
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

    //draw(src_img, "src_img");
    draw(gray_img, "gray_img");

    Mat dft_img, idft_img;

    //傅里叶变换并获取幅度谱
    dft_img = myDFT(gray_img);
    Mat tmp = dft_img.clone();
    swapMat(tmp);
    Mat amplitude_img = getAmplitude(tmp);
    draw(amplitude_img, "amplitude_img");
    //cout << "amplitude_img: " << amplitude_img(Rect(0, 0, 10, 10)) << endl;

    //傅里叶逆变换
    idft_img = myIDFT(dft_img);
    draw(idft_img, "idft_img");

    //理想低通高通滤波器
    gray_img.convertTo(gray_img, CV_32F);
    normalize(gray_img, gray_img, 0, 1, CV_MINMAX);
    double min_val, max_val;
    minMaxLoc(gray_img, &min_val, &max_val);
    Mat ideal_lowpass_img = idealFilter(dft_img, "low", 60);
    Mat ideal_highpass_img = idealFilter(dft_img, "high", 60);
    draw(ideal_lowpass_img, "ideal_lowpass_img");
    draw(ideal_highpass_img, "ideal_highpass_img");
    Mat after_ideal_highpass_img = ideal_highpass_img + gray_img;
    normalize(after_ideal_highpass_img, after_ideal_highpass_img, min_val, max_val, CV_MINMAX);
    draw(after_ideal_highpass_img, "after_ideal_highpass_img");

    //巴特沃斯低通滤波器
    Mat butterworth_low_img = ButterworthFilter(dft_img, 60, 10, "low");
    draw(butterworth_low_img, "butterworth_low_img");

    //巴特沃斯高通滤波器
    Mat butterworth_high_img = ButterworthFilter(dft_img, 60, 10, "high");
    draw(butterworth_high_img, "butterworth_high_img");
    Mat after_butterworth_high_img = butterworth_high_img + gray_img;
    normalize(after_butterworth_high_img, after_butterworth_high_img, min_val, max_val, CV_MINMAX);
    draw(after_butterworth_high_img, "after_butterworth_high_img");
    waitKey(0);
    destroyAllWindows();
    return 0;
}
