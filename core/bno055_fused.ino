#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28);

void quatNormalize(float &w, float &x, float &y, float &z) {
  float n = sqrt(w*w + x*x + y*y + z*z);
  if (n < 1e-12) {
    w = 1.0f;
    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
    return;
  }
  w /= n;
  x /= n;
  y /= n;
  z /= n;
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!bno.begin()) {
    Serial.println("ERR,BNO055_NOT_DETECTED");
    while (1);
  }

  delay(1000);
  bno.setExtCrystalUse(true);
  delay(100);

  Serial.println("READY");
}

void loop() {
  imu::Quaternion q = bno.getQuat();

  float w = q.w();
  float x = q.x();
  float y = q.y();
  float z = q.z();

  quatNormalize(w, x, y, z);

  // Raw fused quaternion from BNO055
  // Format: Q,w,x,y,z
  Serial.print("Q,");
  Serial.print(w, 6); Serial.print(",");
  Serial.print(x, 6); Serial.print(",");
  Serial.print(y, 6); Serial.print(",");
  Serial.println(z, 6);

  delay(20);  // ~50 Hz
}