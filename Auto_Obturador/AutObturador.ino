#include <ESP32Servo.h>
//#define BUTTON D0

Servo myservo;  // crea el objeto servo
 
int pos = 0;    // posicion del servo
volatile int t = 0;
int t0, tf = 0;

int time2rotate = 0; //ms86
/*
ICACHE_RAM_ATTR void cambio() {
  t = millis();
  Serial.print("t: ");
  Serial.println(t0);
}*/

void setup() {
  Serial.begin(9600);
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
   myservo.attach(33);  // vincula el servo al pin digital 9
   myservo.write(170);
   //pinMode(BUTTON, INPUT_PULLUP);
   
   //attachInterrupt(digitalPinToInterrupt(BUTTON), cambio, CHANGE);
   
}

void loop() {
   //varia la posicion de 0 a 180, con esperas de 15ms
  if (Serial.available()>0){
    
    int val = Serial.parseInt() + time2rotate;
    
    myservo.write(110);
    delay(val);
    myservo.write(170);
    
  }
}
