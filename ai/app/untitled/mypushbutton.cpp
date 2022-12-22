#include "mypushbutton.h"

//qizhi::qizhi(QWidget *parent) : QWidget(parent)
//{

//}
Mypushbutton::Mypushbutton(){
//    list.append("background-color: white");
//    list.append("border-style: solid");
//    list.append("border-width:1px");
//    list.append("border-radius:10px");
//    list.append("max-width:20x");
//    list.append("max-height:20px");
//    list.append("min-width:20px");
//    list.append("min-height:20px");
//    list.append("border-color: green");
//    this->setStyleSheet(list.join(';'));
    this->setStyleSheet("QPushButton{background-color:rgba(255,178,0,100%);\
                                                color: white;   border-radius: 10px;  border: 2px groove gray; border-style: outset;}"
            "QPushButton:hover{background-color:white; color: black;}"
            "QPushButton:pressed{background-color:rgb(85, 170, 255); border-style: inset; }"
    );
}



