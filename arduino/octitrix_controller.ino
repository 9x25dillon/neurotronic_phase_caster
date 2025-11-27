/*
 * Octitrix Hardware Controller
 *
 * This Arduino sketch allows hardware control of the Octitrix 8D Audio system.
 * Upload this to your Arduino board to send commands via serial to the Python bridge.
 *
 * Hardware Setup:
 * - Arduino Uno/Nano/Mega (any board with Serial)
 * - Optional: Buttons on pins 2-5 for triggering different patterns
 * - Optional: Potentiometer on A0 for parameter control
 *
 * Commands sent to serial:
 * - "fractal" - Trigger fractal generation
 * - "rhythm" - Trigger euclidean rhythm generation
 * - "dispatch" - Re-send current state
 * - "rhythm:N" - Generate rhythm with N pulses
 *
 * Author: Octitrix Project
 * License: MIT
 */

// Pin definitions
const int BUTTON_FRACTAL = 2;
const int BUTTON_RHYTHM = 3;
const int BUTTON_DISPATCH = 4;
const int BUTTON_RESET = 5;
const int POTENTIOMETER = A0;

// Button state tracking
bool lastFractalState = HIGH;
bool lastRhythmState = HIGH;
bool lastDispatchState = HIGH;
bool lastResetState = HIGH;

// Timing
unsigned long lastAutoTrigger = 0;
const unsigned long AUTO_TRIGGER_INTERVAL = 5000; // 5 seconds

// Mode: 0 = manual, 1 = auto-demo
int mode = 1; // Start in auto-demo mode

void setup() {
  // Initialize serial communication at 9600 baud
  Serial.begin(9600);

  // Set up button pins with internal pull-up resistors
  pinMode(BUTTON_FRACTAL, INPUT_PULLUP);
  pinMode(BUTTON_RHYTHM, INPUT_PULLUP);
  pinMode(BUTTON_DISPATCH, INPUT_PULLUP);
  pinMode(BUTTON_RESET, INPUT_PULLUP);

  // Set up analog input
  pinMode(POTENTIOMETER, INPUT);

  // Wait for serial connection
  delay(1000);

  Serial.println("OCTITRIX:READY");
  Serial.println("OCTITRIX:MODE:AUTO");
}

void loop() {
  // Check for incoming commands from computer
  if (Serial.available() > 0) {
    String incoming = Serial.readStringUntil('\n');
    incoming.trim();

    if (incoming == "ACK") {
      // Acknowledgment from bridge
      // Can add LED feedback here
    } else if (incoming.startsWith("MODE:")) {
      // Mode change command from bridge
      if (incoming.indexOf("MANUAL") >= 0) {
        mode = 0;
        Serial.println("OCTITRIX:MODE:MANUAL");
      } else if (incoming.indexOf("AUTO") >= 0) {
        mode = 1;
        Serial.println("OCTITRIX:MODE:AUTO");
      }
    }
  }

  // Read potentiometer value (0-1023) and map to pulse count (1-16)
  int potValue = analogRead(POTENTIOMETER);
  int pulseCount = map(potValue, 0, 1023, 1, 16);

  if (mode == 1) {
    // AUTO-DEMO MODE
    // Automatically trigger patterns at intervals
    unsigned long currentTime = millis();

    if (currentTime - lastAutoTrigger >= AUTO_TRIGGER_INTERVAL) {
      lastAutoTrigger = currentTime;

      // Alternate between fractal and rhythm
      static bool togglePattern = false;

      if (togglePattern) {
        Serial.println("CMD:fractal");
      } else {
        // Use potentiometer value for rhythm pulses
        Serial.print("CMD:rhythm:");
        Serial.println(pulseCount);
      }

      togglePattern = !togglePattern;
    }

  } else {
    // MANUAL MODE
    // Check button presses

    // Fractal button
    bool fractalPressed = digitalRead(BUTTON_FRACTAL);
    if (fractalPressed == LOW && lastFractalState == HIGH) {
      Serial.println("CMD:fractal");
      delay(50); // Debounce
    }
    lastFractalState = fractalPressed;

    // Rhythm button
    bool rhythmPressed = digitalRead(BUTTON_RHYTHM);
    if (rhythmPressed == LOW && lastRhythmState == HIGH) {
      // Send rhythm command with pulse count from potentiometer
      Serial.print("CMD:rhythm:");
      Serial.println(pulseCount);
      delay(50); // Debounce
    }
    lastRhythmState = rhythmPressed;

    // Dispatch button
    bool dispatchPressed = digitalRead(BUTTON_DISPATCH);
    if (dispatchPressed == LOW && lastDispatchState == HIGH) {
      Serial.println("CMD:dispatch");
      delay(50); // Debounce
    }
    lastDispatchState = dispatchPressed;

    // Reset button
    bool resetPressed = digitalRead(BUTTON_RESET);
    if (resetPressed == LOW && lastResetState == HIGH) {
      Serial.println("CMD:reset");
      delay(50); // Debounce
    }
    lastResetState = resetPressed;
  }

  // Small delay to prevent overwhelming the serial buffer
  delay(10);
}
