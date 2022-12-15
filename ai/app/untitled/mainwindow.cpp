#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "mypushbutton.h"
#include <QtWidgets>
#include <QVideosurfaceFormat>
#include <QPushButton>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{

    ui->setupUi(this);
    setFixedSize(1200,800);
    camera_ = new QCamera;
    surface_ = new MyVideoSurface(this);
    camera_->setViewfinder(surface_);
    camera_->start();//这里启动
    connect(ui->pushButton,&QPushButton::clicked,[=](){
       QString str = surface_->savejpg();
        ui->label->setText(str);
    });
    connect(ui->pushButton_2,&QPushButton::clicked,[=](){
       QString str = surface_->train();
        ui->label->setText(str);
    });
//    Mypushbutton * start2 = new Mypushbutton();
//    start2->setParent(this);
//    start2->move(1000,700);
//    connect(start2,&Mypushbutton::clicked,[=](){
//        surface_->savejpg();
//    });
}

MainWindow::~MainWindow()
{
    delete ui;
}

QSize MainWindow::sizeHint() const
{
    return surface_->surfaceFormat().sizeHint();
}

void MainWindow::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    if (surface_->isActive()) {
        const QRect videoRect = surface_->videoRect();
        if (!videoRect.contains(event->rect())) {
            QRegion region = event->region();
            region = region.subtracted(videoRect);
            QBrush brush = palette().window();
            for (const QRect &rect : region){
                painter.fillRect(rect, brush);
            }
        }
        surface_->paint(&painter);//在主窗口绘制
    } else {
        painter.fillRect(event->rect(), palette().window());
    }
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
    surface_->updateVideoRect();
}
