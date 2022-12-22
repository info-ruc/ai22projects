#ifndef MYPUSHBUTTON_H
#define MYPUSHBUTTON_H

#include <QWidget>
#include <QPushButton>

class Mypushbutton : public QPushButton
{
    Q_OBJECT
public:
//    explicit Mypushbutton(QWidget *parent = nullptr);
    QStringList list;
    Mypushbutton();

signals:

};

#endif // MYPUSHBUTTON_H
