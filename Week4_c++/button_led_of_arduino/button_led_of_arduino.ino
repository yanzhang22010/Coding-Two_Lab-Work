const int buttonPin = 2;  // 按钮引脚
const int ledPin = 13;    // LED 引脚

int buttonState = 0;      // 按钮状态

void setup() {
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  // 读取按钮状态
  buttonState = digitalRead(buttonPin);

  // 如果按钮按下
  if (buttonState == LOW) {
    digitalWrite(ledPin, LOW);  // 熄灭 LED
    Serial.write('0');          // 向电脑发送 0
  } else {
    digitalWrite(ledPin, HIGH); // 点亮 LED
    Serial.write('1');          // 向电脑发送 1
  }

  delay(50);  // 延迟 50 毫秒
}
