#include<LiquidCrystal.h>
#include <Servo.h>


//Humedad
int humedad = 0;
int h_p = 0;
const int SensorHumedad = A0;
const int AirValue = 1023;
const int WaterValue = 200;


//Luz
int luz = 0;
int l_p = 0;
const int SensorLuz = A1;


//Servo
int pos = 0;
Servo servo_9;


//LCD
LiquidCrystal lcd(3, 2, 4, 5, 6, 7);


//Pulsador
int boton = 0;
int flag = 0;
int cont = 0;
const int Pulsador = 10; 


//PLanta
String nombre = "MiPlanta1";
//Bomba
const int driver_ENA;


//Serial
String cad = "Especie";
/////////////////////

void setup()
{
  pinMode(A0, INPUT);
  pinMode(A1, INPUT);

  //Bomba  cafe(azul_3) rojo
  pinMode(12, OUTPUT);
  pinMode(13, OUTPUT);
 
  
  //LCD
  lcd.begin(16, 2);
  
  //Sevo
  servo_9.attach(9, 500, 2500);
  
  //Serial
  Serial.begin(9600);

  //Pulsador
  pinMode(Pulsador,INPUT);
}

void loop()
{
  humedad = analogRead(SensorHumedad);
  luz = analogRead(SensorLuz);

  h_p = map(humedad, AirValue, WaterValue, 0, 100);
  l_p = int((double(luz)/550)*100);

  flag = boton;
  boton=digitalRead(Pulsador);
  
  Serial.println(humedad);
  Serial.println(luz);
  //Serial.println(boton);

  switch (cont){
    case 0:
      lcd.clear();
      lcd.setCursor(0,0);
      lcd.print(nombre);
      lcd.setCursor(0,1);
      lcd.print(cad);
      break;

    case 1:
      lcd.clear();
      lcd.setCursor(0,0);
      lcd.print("Humedad: " + String(humedad));
      lcd.setCursor(0,1);
      lcd.print("Luz: " + String(luz));
      break;
    
    case 2:
      lcd.clear();
      lcd.setCursor(0,0);
      lcd.print("Humedad: " + String(h_p));
      lcd.setCursor(0,1);
      lcd.print("Luz: " + String(l_p));
      break;
   }
   
  if (boton != flag){
    if(cont == 2)
      cont = 0;
      
    else
      cont++;
  }

  if (Serial.available() > 0) {
    // read the incoming byte:
    cad = Serial.readString();

    // say what you got:
    Serial.print("I received: ");
    Serial.println(cad);
  }

  digitalWrite(12, LOW);
  digitalWrite(13, LOW);

  if (h_p > 0 && h_p < 10){
    digitalWrite(12, HIGH);
    digitalWrite(13, HIGH);
    //analogWrite(driver_ENA, 10); //Velocidad
  }
  
  //if (h_p <= 2)
    //digitalWrite(11, HIGH);

  //else if (h_p >= 80)
   // digitalWrite(11, LOW);


  //if 
  
  delay(300);
}
