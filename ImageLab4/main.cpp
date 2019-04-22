#include <iostream>
#include <opencv2/opencv.hpp>
#include<vector>
using namespace std;
using namespace cv;

string imageDir = "/home/jj/Pictures/";

string imageName = "lena.png";

void draw(Mat img, string name) {
    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name,img);
}

Mat addSalt(Mat src_img, float per=0.1) {
    Mat result_img = src_img.clone();
    int N = (int)(per * src_img.rows * src_img.cols);
    for(int k=0; k<=N; k++) {
        int i = (int)(rand()*1.0/RAND_MAX * src_img.cols);
        int j = (int)(rand()*1.0/RAND_MAX * src_img.rows);
        if(src_img.channels() == 1) result_img.at<uchar>(j, i) = 255;
        else {
            Mat_<Vec3b> _img = result_img;
            for(int c=0; c<3; c++) {
                _img(j, i)[c] = 255;
            }
        }
    }
    return result_img;
}
Mat addPepper(Mat src_img, float per=0.1) {
    Mat result_img = src_img.clone();
    int N = (int)(per * src_img.rows * src_img.cols);
    for(int k=0; k<=N; k++) {
        int i = (int)(rand()*1.0/RAND_MAX * src_img.cols);
        int j = (int)(rand()*1.0/RAND_MAX * src_img.rows);
        if(src_img.channels() == 1) result_img.at<uchar>(j, i) = 0;
        else {
            Mat_<Vec3b> _img = result_img;
            for(int c=0; c<3; c++) {
                _img(j, i)[c] = 0;
            }
        }
    }
    return result_img;
}
Mat addPepSalt(Mat src_img, float per=0.1) {
    Mat result_img = src_img.clone();
    int N = (int)(per * src_img.rows * src_img.cols);
    for(int k=0; k<=N; k++) {
        int val;
        int i = (int)(rand()*1.0/RAND_MAX * src_img.cols);
        int j = (int)(rand()*1.0/RAND_MAX * src_img.rows);
        if(rand() & 1) val = 255;
        else val = 0;
        if(src_img.channels() == 1) result_img.at<uchar>(j, i) = val;
        else {
            Mat_<Vec3b> _img = result_img;
            for(int c=0; c<3; c++) {
                _img(j, i)[c] = val;
            }
        }
    }
    return result_img;
}

double generateGaussianNoise()
{
    static bool hasSpare = false;
    static double rand1, rand2;

    if(hasSpare)
    {
        hasSpare = false;
        return sqrt(rand1) * sin(rand2);
    }

    hasSpare = true;

    rand1 = rand() / ((double) RAND_MAX);
    if(rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((double) RAND_MAX)) * 2 * CV_PI;

    return sqrt(rand1) * cos(rand2);
}
Mat addGauss(Mat src_img, int k= 32) {
    Mat result_img = src_img.clone();
    int channels = result_img.channels();
    int nRows = result_img.rows;
    int nCols = result_img.cols * channels;
    if(result_img.isContinuous()){
        nCols *= nRows;
        nRows = 1;
    }
    //cout << result_img.row(0) << endl;
    for(int i=0; i<nRows; i++) {
        uchar* p = result_img.ptr<uchar>(i);
        for(int j=0; j<nCols; j++) {
           float val = p[j] + k * generateGaussianNoise();
           if(val > 255) val = 255;
           else if(val < 0) val = 0;
           p[j] = (uchar)val;
        }
    }
        //cout << result_img << endl;
    return result_img;
}

//1、均值滤波
//具体内容:利用 OpenCV 对灰度图像像素进行操作,分别利用算术均值滤
//波器、几何均值滤波器、谐波和逆谐波均值滤波器进行图像去噪。模板大小为
//5*5。
//(注:请分别为图像添加高斯噪声、胡椒噪声、盐噪声和椒盐噪声,并观察
//滤波效果)
//算数均值
Mat meanFilter(Mat src_img, int k_size=5) {
    Mat result_img;
    Mat kernel = Mat::ones(k_size, k_size, CV_32F);
    kernel /= (float)(k_size * k_size);
    filter2D(src_img, result_img, src_img.depth(), kernel);
    return result_img;
}
//几何均值
double geoCounter(Mat src) {
    double geo = 1;
    for(int i=0; i<src.rows; i++) {
        uchar* data = src.ptr<uchar>(i);
        for(int j=0; j<src.cols; j++) {
            if(data[j] != 0)
                geo *= data[i];
        }
    }
    return pow(geo, 1.0/(src.rows * src.cols));
}
Mat geoMeanFilter(Mat src_img, int k_size=5) {
    Mat result_img =Mat(src_img.rows, src_img.cols, src_img.type());
    Mat channels[3];
    if(src_img.channels() == 3) {
        split(src_img, channels);
    }
    int width = k_size/2;
    for(int i=width; i<src_img.rows-width; i++) {
        for(int j=width; j<src_img.cols-width; j++) {
            if(src_img.channels() == 3) {
                Mat_<Vec3b> _img = result_img;
                for(int c=0; c<3; c++) {
                    _img(i, j)[c] = saturate_cast<uchar>(geoCounter(
                                        channels[c](Rect(j-width, i-width, k_size, k_size))));
                }
            }
            else {
                result_img.at<uchar>(i, j) = saturate_cast<uchar>(geoCounter(
                            src_img(Rect(j-width, i-width, k_size, k_size))));
            }
        }
    }
    return result_img;
}
//谐波均值
double harmonicCounter(Mat src) {
    double harmon = 0;
    for(int i=0; i<src.rows; i++) {
        uchar* data = src.ptr<uchar>(i);
        for(int j=0; j<src.cols; j++) {
            if(data[j] != 0)
                harmon += 1.0/data[i];
        }
    }
    return (src.rows * src.cols / harmon);
}
Mat harmonicFilter(Mat src_img, int k_size=5) {
    Mat result_img =Mat(src_img.rows, src_img.cols, src_img.type());
    Mat channels[3];
    if(src_img.channels() == 3) {
        split(src_img, channels);
    }
    int width = k_size/2;
    for(int i=width; i<src_img.rows-width; i++) {
        for(int j=width; j<src_img.cols-width; j++) {
            if(src_img.channels() == 3) {
                Mat_<Vec3b> _img = result_img;
                for(int c=0; c<3; c++) {
                    _img(i, j)[c] = saturate_cast<uchar>(harmonicCounter(
                                        channels[c](Rect(j-width, i-width, k_size, k_size))));
                }
            }
            else {
                result_img.at<uchar>(i, j) = saturate_cast<uchar>(harmonicCounter(
                            src_img(Rect(j-width, i-width, k_size, k_size))));
            }
        }
    }
    return result_img;
}
//逆谐波均值
double inHarmonicCount(Mat src, double q) {
    double inharm_up = 0;
    double inharm_down = 0;
    for(int i=0; i<src.rows; i++) {
        uchar* data = src.ptr<uchar>(i);
        for(int j=0; j<src.cols; j++) {
            double val = pow(data[i], q);
            inharm_up += val * data[j];
            inharm_down += val;
        }
    }
    return inharm_up / inharm_down;
}
Mat inHarmonicFilter(Mat src_img, double q=1.0, int k_size=5) {
    Mat result_img =Mat(src_img.rows, src_img.cols, src_img.type());
    Mat channels[3];
    if(src_img.channels() == 3) {
        split(src_img, channels);
    }
    int width = k_size/2;
    for(int i=width; i<src_img.rows-width; i++) {
        for(int j=width; j<src_img.cols-width; j++) {
            if(src_img.channels() == 3) {
                Mat_<Vec3b> _img = result_img;
                for(int c=0; c<3; c++) {
                    _img(i, j)[c] =
                            saturate_cast<uchar>(inHarmonicCount(channels[c](
                                                                 Rect(j-width, i-width,
                                                                      k_size, k_size)), q));
                }
            }
            else {
                result_img.at<uchar>(i, j) = saturate_cast<uchar>(inHarmonicCount(
                            src_img(Rect(j-width, i-width, k_size, k_size)), q));
            }
        }
    }
    return result_img;
}

//2、中值滤波
//具体内容:利用 OpenCV 对灰度图像像素进行操作,分别利用 5*5 和 9*9
//尺寸的模板对图像进行中值滤波。(注:请分别为图像添加胡椒噪声、盐噪声和
//椒盐噪声,并观察滤波效果)
uchar quicksort(uchar* buffer, int position, int l, int r) {
    if(l >= r) return buffer[l];
    uchar temp = buffer[l];
    int nl = l;
    int nr = r;
    while(nl < nr) {
        uchar tmp;
        while(nl < nr && temp <= buffer[nr])
            nr--;
        tmp = buffer[nr];
        buffer[nr] = buffer[nl];
        buffer[nl] = tmp;
        while(nl < nr && buffer[nl] <= temp) nl++;
        tmp = buffer[nr];
        buffer[nr] = buffer[nl];
        buffer[nl] = tmp;
    }
    buffer[nl] = temp;
    if(position == nl) return temp;
    else if(nl > position)
        return quicksort(buffer, position, l, nl-1);
    else
        return quicksort(buffer, position, nl+1, r);
}
uchar midCounter(Mat src) {
    int len = src.rows * src.cols;
    int pos;
    int index = 0;
    uchar* buffer = new uchar[len];
    for(int i=0; i<src.rows; i++) {
        uchar* data = src.ptr<uchar>(i);
        for(int j=0; j<src.cols; j++) {
            buffer[index++] = data[j];
        }
    }
    uchar val = quicksort(buffer, len/2, 0, len-1);

    return quicksort(buffer, len/2, 0, len-1);
}
Mat midFilter(Mat src_img, int k_size=5) {
    Mat result_img =Mat(src_img.rows, src_img.cols, src_img.type());
    Mat channels[3];
    if(src_img.channels() == 3) {
        split(src_img, channels);
    }
    int width = k_size/2;
    for(int i=width; i<src_img.rows-width; i++) {
        for(int j=width; j<src_img.cols-width; j++) {
            if(src_img.channels() == 3) {
                Mat_<Vec3b> _img = result_img;
                for(int c=0; c<3; c++) {
                    _img(i, j)[c] =
                            saturate_cast<uchar>(midCounter(channels[c](
                                                                 Rect(j-width, i-width,
                                                                      k_size, k_size))));
                }
            }
            else {
                result_img.at<uchar>(i, j) = saturate_cast<uchar>(midCounter(
                            src_img(Rect(j-width, i-width, k_size, k_size))));
            }
        }
    }
    return result_img;
}
//3、自适应均值滤波。
//具体内容:利用 OpenCV 对灰度图像像素进行操作,设计自适应局部降
//低噪声滤波器去噪算法。模板大小 7*7 (对比该算法的效果和均值滤波器的效果)
uchar modifyMeanCounter(Mat src, int now_size, int max_size) {
    //Pxy是当前点灰度值， Pmax，Pmin分别是模板内最大灰度值和最小值，Pmean是模板内灰度均值
    uchar Pxy = src.at<uchar>(now_size/2, now_size/2);
    uchar Pmax = 0;
    uchar Pmin = 255;
    uchar Pmean;
    int sum = 0;
    for(int i=0; i<now_size; i++) {
        uchar* data = src.ptr<uchar>(i);
        for(int j=0; j<now_size; j++) {
            sum += data[j];
            if(data[j] < Pmin) Pmin = data[j];
            else if(Pmax < data[j]) Pmax = data[j];
        }
    }
    Pmean = saturate_cast<uchar>( sum / (now_size * now_size) );
    //如果Pmean在最大最小灰度之间（噪声较少）
    if(Pmean < Pmean && Pmean < Pmax) {
        //如果当前像素在最大最小值之间（即不是噪声）返回Pxy, 否则返回Pmean
        if(Pmin < Pxy && Pxy < Pmax)
            return Pxy;
        else
            return Pmean;
    }
    //噪声较多，增大模板尺寸
    else {
        now_size += 2;
        if(now_size <= max_size)
            return modifyMeanCounter(src, now_size, max_size);
        else
            return Pmean;
    }
}
Mat modifyMeanFilter(Mat src_img, int min_size=3, int max_size=7) {
    int width = max_size/2;
    Mat result_img, temp_img;
    result_img = Mat(src_img.rows + max_size, src_img.cols + max_size, src_img.type());
    copyMakeBorder(src_img, temp_img, width, width, width, width, BORDER_REFLECT);
    Mat channels[3];
    if(src_img.channels() == 3) {
        split(temp_img, channels);
    }

    for(int i=width; i<temp_img.rows-width; i++) {
        for(int j=width; j<temp_img.cols-width; j++) {
            if(src_img.channels() == 3) {
                Mat_<Vec3b> _img = result_img;
                for(int c=0; c<3; c++) {
                    _img(i, j)[c] = saturate_cast<uchar>(
                                modifyMeanCounter(channels[c](
                                                 Rect(j-width, i-width,
                                                      max_size, max_size)),
                                        min_size, max_size));
                }
            }
            else {
                result_img.at<uchar>(i, j) = saturate_cast<uchar>(modifyMeanCounter(
                            temp_img(Rect(j-width, i-width, max_size, max_size)),
                                                                      min_size, max_size));
            }
        }
    }
    return result_img(Rect(width, width, src_img.cols, src_img.rows));
}
//4、自适应中值滤波
//具体内容:利用 OpenCV 对灰度图像像素进行操作,设计自适应中值滤波算
//法对椒盐图像进行去噪。模板大小 7*7(对比中值滤波器的效果)
uchar modifyMidCounter(Mat src, int now_size, int max_size) {
    vector<uchar> buffer;
    for(int i=0; i<now_size; i++) {
        uchar* data = src.ptr<uchar>(i);
        for(int j=0; j<now_size; j++) {
            buffer.push_back(data[j]);
        }
    }
    sort(buffer.begin(), buffer.end());
    uchar Pxy = src.at<uchar>(now_size/2, now_size/2);
    uchar Pmax = *(buffer.end() - 1);
    uchar Pmin = buffer[0];
    uchar Pmid = buffer[buffer.size() / 2];
    if(Pmin < Pmid && Pmid < Pmax) {
        if(Pmin < Pxy && Pxy < Pmax)
            return Pxy;
        else
            return Pmid;
    }
    else {
        now_size += 2;
        if(now_size <= max_size)
            return modifyMidCounter(src, now_size, max_size);
        else
            return Pmid;
    }
}
Mat modifyMidFilter(Mat src_img, int min_size=3, int max_size=7) {
    int width = max_size/2;
    Mat result_img, temp_img;
    result_img = Mat(src_img.rows + max_size, src_img.cols + max_size, src_img.type());
    copyMakeBorder(src_img, temp_img, width, width, width, width, BORDER_REFLECT);
    Mat channels[3];
    if(src_img.channels() == 3) {
        split(temp_img, channels);
    }

    for(int i=width; i<temp_img.rows-width; i++) {
        for(int j=width; j<temp_img.cols-width; j++) {
            if(src_img.channels() == 3) {
                Mat_<Vec3b> _img = result_img;
                for(int c=0; c<3; c++) {
                    _img(i, j)[c] = saturate_cast<uchar>(
                                modifyMidCounter(channels[c](
                                                 Rect(j-width, i-width,
                                                      max_size, max_size)),
                                        min_size, max_size));
                }
            }
            else {
                result_img.at<uchar>(i, j) = saturate_cast<uchar>(modifyMidCounter(
                            temp_img(Rect(j-width, i-width, max_size, max_size)),
                                                                      min_size, max_size));
            }
        }
    }
    return result_img(Rect(width, width, src_img.cols, src_img.rows));
}
//5、彩色图像均值滤波
//具体内容:利用 OpenCV 对彩色图像 RGB 三个通道的像素进行操作,利用算
//术均值滤波器和几何均值滤波器进行彩色图像去噪。模板大小为 5*5。

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


//    draw(gray_img, "gray_img");
//    Mat gauss_noise_img = addGauss(gray_img);
//    Mat pepper_noise_img = addPepper(gray_img);
//    Mat salt_noise_img = addSalt(gray_img);
//    Mat pepsalt_noise_img = addPepSalt(gray_img);
//    draw(salt_noise_img, "salt");
//    draw(pepper_noise_img, "pepper");
//    draw(pepsalt_noise_img, "pepsalt");
//    draw(gauss_noise_img, "gause");

    draw(src_img, "src_img");
    draw(gray_img, "gray_img");
    Mat gauss_noise_RGBimg = addGauss(src_img);
    Mat pepper_noise_RGBimg = addPepper(src_img);
    Mat salt_noise_RGBimg = addSalt(src_img);
    Mat pepsalt_noise_RGBimg = addPepSalt(src_img);
    draw(gauss_noise_RGBimg, "salt");
    draw(pepper_noise_RGBimg, "pepper");
    draw(salt_noise_RGBimg, "pepsalt");
    draw(pepsalt_noise_RGBimg, "gause");

//    //算数均值
//    Mat g_mean_img = meanFilter(gauss_noise_img);
//    Mat p_mean_img = meanFilter(pepper_noise_img);
//    Mat s_mean_img = meanFilter(salt_noise_img);
//    Mat ps_mean_img = meanFilter(pepsalt_noise_img);
//    draw(g_mean_img, "g_mean_img");
//    draw(p_mean_img, "p_mean_img");
//    draw(s_mean_img, "s_mean_img");
//    draw(ps_mean_img, "ps_mean_img");


//    //几何均值
//    Mat g_geo_img = geoMeanFilter(gauss_noise_img);
//    Mat p_geo_img = geoMeanFilter(pepper_noise_img);
//    Mat s_geo_img = geoMeanFilter(salt_noise_img);
//    Mat ps_geo_img = geoMeanFilter(pepsalt_noise_img);
//    draw(g_geo_img, "g_geo_img");
//    draw(p_geo_img, "p_geo_img");
//    draw(s_geo_img, "s_geo_img");
//    draw(ps_geo_img, "ps_geo_img");

//    //谐波均值
//    Mat g_harmon_img = harmonicFilter(gauss_noise_img);
//    Mat p_harmon_img = harmonicFilter(pepper_noise_img);
//    Mat s_harmon_img = harmonicFilter(salt_noise_img);
//    Mat ps_harmon_img = harmonicFilter(pepsalt_noise_img);
//    draw(g_harmon_img, "g_harmon_img");
//    draw(p_harmon_img, "p_harmon_img");
//    draw(s_harmon_img, "s_harmon_img");
//    draw(ps_harmon_img, "ps_harmon_img");

//    //逆谐波均值
//    Mat g_inharmon_img = inHarmonicFilter(gauss_noise_img);
//    Mat p_inharmon_img = inHarmonicFilter(pepper_noise_img);
//    Mat s_inharmon_img = inHarmonicFilter(salt_noise_img);
//    Mat ps_inharmon_img = inHarmonicFilter(pepsalt_noise_img);
//    draw(g_inharmon_img, "g_inharmon_img");
//    draw(p_inharmon_img, "p_inharmon_img");
//    draw(s_inharmon_img, "s_inharmon_img");
//    draw(ps_inharmon_img, "ps_inharmon_img");

//    //中值滤波
//    //Mat g_mid_img = midFilter(gauss_noise_img);
//    Mat p_mid_img = midFilter(pepper_noise_img, 9);
//    Mat s_mid_img = midFilter(salt_noise_img, 9);
//    Mat ps_mid_img = midFilter(pepsalt_noise_img, 9);
//    //draw(g_mid_img, "g_mid_img");
//    draw(p_mid_img, "p_mid_img");
//    draw(s_mid_img, "s_mid_img");
//    draw(ps_mid_img, "ps_mid_img");

//    //自适应均值滤波
//    Mat g_modifyMean_img = modifyMeanFilter(gauss_noise_img);
//    Mat p_modifyMean_img = modifyMeanFilter(pepper_noise_img);
//    Mat s_modifyMean_img = modifyMeanFilter(salt_noise_img);
//    Mat ps_modifyMean_img = modifyMeanFilter(pepsalt_noise_img);
//    draw(g_modifyMean_img, "g_modifyMean_img");
//    draw(p_modifyMean_img, "p_modifyMean_img");
//    draw(s_modifyMean_img, "s_modifyMean_img");
//    draw(ps_modifyMean_img, "ps_modifyMean_img");

//    //自适应中值滤波
////    Mat g_modifyMid_img = modifyMidFilter(gauss_noise_img);
//    Mat p_modifyMid_img = modifyMidFilter(pepper_noise_img);
//    Mat s_modifyMid_img = modifyMidFilter(salt_noise_img);
//    Mat ps_modifyMid_img = modifyMidFilter(pepsalt_noise_img);
////    draw(g_modifyMid_img, "g_modifyMid_img");
//    draw(p_modifyMid_img, "p_modifyMid_img");
//    draw(s_modifyMid_img, "s_modifyMid_img");
//    draw(ps_modifyMid_img, "ps_modifyMid_img");


//    //彩色均值滤波
//    Mat g_mean_img = meanFilter(gauss_noise_RGBimg);
//    Mat p_mean_img = meanFilter(pepper_noise_RGBimg);
//    Mat s_mean_img = meanFilter(salt_noise_RGBimg);
//    Mat ps_mean_img = meanFilter(pepsalt_noise_RGBimg);
//    draw(g_mean_img, "g_mean_img");
//    draw(p_mean_img, "p_mean_img");
//    draw(s_mean_img, "s_mean_img");
//    draw(ps_mean_img, "ps_mean_img");

    Mat g_geomean_img = geoMeanFilter(gauss_noise_RGBimg);
    Mat p_geomean_img = geoMeanFilter(pepper_noise_RGBimg);
    Mat s_geomean_img = geoMeanFilter(salt_noise_RGBimg);
    Mat ps_geomean_img = geoMeanFilter(pepsalt_noise_RGBimg);
    draw(g_geomean_img, "g_geomean_img");
    draw(p_geomean_img, "p_geomean_img");
    draw(s_geomean_img, "s_geomean_img");
    draw(ps_geomean_img, "ps_geomean_img");
    waitKey(0);




    destroyAllWindows();
    return 0;
}
