TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
INCLUDEPATH+=/home/jj/opencv/include\
            /home/jj/opencv/include\opencv\
            /home/jj/opencv/include\opencv2
LIBS+=-L/home/jj/opencv/release/lib\
        /home/jj/Tools/opencv/release/lib/libopencv*
SOURCES += main.cpp
