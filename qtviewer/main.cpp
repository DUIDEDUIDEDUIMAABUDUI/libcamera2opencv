#include "window.h"
#include "fatigue_detector.h"

#include <QImage>
#include <QPixmap>
#include <QColor>

FatigueDetector detector; 

Window::Window()
{
    myCallback.window = this;
    camera.registerCallback(&myCallback);

   
    thermo = new QwtThermo; 
    thermo->setFillBrush(QBrush(Qt::red));
    thermo->setScale(0, 255);
    thermo->show();

    image = new QLabel;


    hLayout = new QHBoxLayout();
    hLayout->addWidget(thermo);
    hLayout->addWidget(image);

    setLayout(hLayout);

    camera.start();
}

Window::~Window()
{
    camera.stop();
}

void Window::updateImage(const cv::Mat &mat) {
    cv::Mat output;
    bool drowsy = detector.detect(mat, output); 

  
    const QImage frame(output.data, output.cols, output.rows, output.step, QImage::Format_RGB888);

    image->setPixmap(QPixmap::fromImage(frame));

    
    const int h = frame.height();
    const int w = frame.width();
    const QColor c = frame.pixelColor(w / 2, h / 2);
    thermo->setValue(c.lightness());

    update();
}
