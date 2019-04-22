#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

string imageDir = "/home/jj/Pictures/";

void draw(Mat img, string name) {
    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name,img);
}

Mat bin_trans(Mat src_img, int val_thresh) {
    if(src_img.empty())
        std::cout<< "No data!" <<std::endl;
    Mat result_img;
    threshold(src_img, result_img, val_thresh, 255, CV_THRESH_BINARY);
    return result_img;
}

Mat my_bin_trains(Mat src_img, float val_thresh) {
    Mat result_img = src_img.clone();
    for(int i=0; i<result_img.rows; i++) {
        uchar* datas = result_img.ptr<uchar>(i);
        for(int j=0; j<result_img.cols; j++) {
            if(datas[j] < val_thresh)
                datas[j] = 0;
            else datas[j] = 255;
        }
    }
    return result_img;
}

Mat log_trans(Mat src_img, float c=1) {
    if(src_img.empty())
        cout<< "No data!" <<endl;
    Mat result_img;
    result_img = src_img + 1;

    result_img.convertTo(result_img, CV_32F);
    log(result_img, result_img);
    result_img *= c;
    normalize(result_img, result_img, 0, 255, NORM_MINMAX);
    result_img.convertTo(result_img, CV_8U);
    return result_img;
}

Mat gama_trans(Mat src_img, float gama=1.0, float c=1.0) {
    if(src_img.empty())
        cout<< "No data!" <<endl;
    Mat result_img;
    src_img.convertTo(result_img, CV_32F);
    normalize(result_img, result_img, 0, 1, NORM_MINMAX);
    cv::pow(result_img, gama, result_img);
    result_img *= c;
    normalize(result_img, result_img, 0, 255, NORM_MINMAX);
    result_img.convertTo(result_img, CV_8U);
    return result_img;
}

Mat comp_trans1(Mat src_img) {
    Mat hsv_img, result_img;
    src_img.convertTo(hsv_img, CV_32F);
    cvtColor(hsv_img, hsv_img, CV_RGB2HSV);
    for(int i=0; i<hsv_img.rows; i++) {
        for(int j=0; j<hsv_img.cols; j++) {
            float val = hsv_img.at<Vec3f>(i, j)[0] + 180;
            if(val > 360) val -= 360;
            hsv_img.at<Vec3f>(i, j)[0] = val;
        }
    }
    cvtColor(hsv_img, hsv_img, CV_HSV2RGB);
    hsv_img.convertTo(result_img, CV_8U);
    return result_img;
}

Mat comp_trans2(Mat src_img) {
    Mat result_img = src_img.clone();
    Mat channels[3];
    split(result_img, channels);

    for(int i=0; i<result_img.rows; i++) {
        for(int j=0; j<result_img.cols; j++) {
            //当前点RGB值最小最大值
            double min_val, max_val;
            min_val = channels[0].at<uchar>(i, j);
            max_val = min_val;
            for(int c=1; c<3; c++) {
                double temp  = channels[c].at<uchar>(i, j);
               if(temp > max_val) max_val = temp;
               if(temp < min_val) min_val = temp;
            }
            Mat_<Vec3b> _img = result_img;
            //Sij(r, g , b) = (max(Rij(r, g, b)) + min(Rij(r, g, b))) * zeros(3) - Rij(r, g, b)
            for(int c=0; c<3; c++) {
                double val = min_val + max_val - _img(i, j)[c];
                if(val > 255) val = 255;
                _img(i, j)[c] = saturate_cast<uchar>(val);
            }
        }
    }
    return result_img;
}

int main()
{    
    cout<<"please input a image file name:"<<endl;
    string imageName = "lena.png";
    //cin>>imageName;
    string filePath=imageDir + imageName;
    cout<<"The input is "<<filePath<<endl;

    Mat src_img=imread(filePath), gray_img;
    if(src_img.empty())
    {
        cout<<"image is empty!"<<endl;
    }


    cvtColor(src_img, gray_img, COLOR_BGR2GRAY);

//    //二值变换
//    draw(src_img, "src_img");
//    draw(gray_img, "gray_img");
//    waitKey(0);
//    for(int i=80; i<=120; i+=10) {
//        Mat bin_img = bin_trans(gray_img, i);
//        stringstream name;
//        name << "bin_img" << i;
//        draw(bin_img, name.str());
//    }
//    waitKey(0);
//    destroyAllWindows();

//    //log变换
//    draw(src_img, "src_img");
//    draw(gray_img, "gray_img");
//    waitKey(0);
//    Mat log_img = log_trans(gray_img, 1);
//    draw(log_img, "log_img");
//    waitKey(0);
//    destroyAllWindows();

//    //gama变换
//    draw(src_img, "src_img");
//    draw(gray_img, "gray_img");
//    waitKey(0);
//    //gama > 1
//    for(int i=1; i<=25; i+=3) {
//        stringstream name;
//        name << "gama = " << i;
//        Mat gama_img = gama_trans(gray_img, i);
//        draw(gama_img, name.str());
//    }
//    waitKey(0);
//    destroyAllWindows();
//    draw(src_img, "src_img");
//    draw(gray_img, "gray_img");
//    waitKey(0);
//    //gama < 1
//    for(double i=0.67; i>0; i-=0.2) {
//        stringstream name;
//        name << "gama = " << i;
//        Mat gama_img = gama_trans(gray_img, i);
//        draw(gama_img, name.str());
//    }
//    waitKey(0);
//    destroyAllWindows();

    //补色变换
    draw(src_img, "src_img");
    draw(gray_img, "gray_img");
    waitKey(0);
    Mat comp_img1 = comp_trans1(src_img);
    draw(comp_img1, "comp_img1");
    Mat comp_img2 = comp_trans2(src_img);
    draw(comp_img2, "comp_img2");

    draw(comp_img1 - comp_img2, "de_img");
    cout << format(comp_img1 - comp_img2, Formatter::FMT_PYTHON) << endl;
    waitKey(0);
    destroyAllWindows();

    return 0;
}
