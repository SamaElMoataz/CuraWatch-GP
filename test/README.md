# Smart Medical Watch — Complete Codebase

## File Structure

```
esp32_medwatch/
└── main.ino                ← ESP32 firmware (Arduino IDE)

phone_app/
├── signal_processing.py   ← PPG + ECG preprocessing and feature extraction
├── inference.py           ← ML inference engine + calibrator
├── train_models.py        ← Train and save models (run once)
├── server.py              ← Flask API server (receives data from ESP32)
└── models/                ← Auto-created by train_models.py
    ├── rf_bp_model.pkl
    ├── glucose_model.pkl
    ├── scaler_X.pkl
    ├── scaler_SBP.pkl
    ├── scaler_DBP.pkl
    ├── scaler_glu_X.pkl
    └── scaler_glu_y.pkl
```

---

## ESP32 Wiring

| Sensor     | Sensor Pin | ESP32 Pin |
|------------|-----------|-----------|
| MAX30102   | SDA       | GPIO 21   |
| MAX30102   | SCL       | GPIO 22   |
| MAX30102   | VIN       | 3.3V      |
| MAX30102   | GND       | GND       |
| AD8232     | OUTPUT    | GPIO 34   |
| AD8232     | LO+       | GPIO 32   |
| AD8232     | LO-       | GPIO 33   |
| AD8232     | VCC       | 3.3V      |
| AD8232     | GND       | GND       |
| MPU6050    | SDA       | GPIO 21   |
| MPU6050    | SCL       | GPIO 22   |
| MPU6050    | VCC       | 3.3V      |
| MPU6050    | GND       | GND       |

MAX30102 and MPU6050 share the same I2C bus (SDA + SCL).

---

## Arduino Libraries Required

Install via Arduino Library Manager:
- `SparkFun MAX3010x Pulse and Proximity Sensor Library`
- `ArduinoJson` (version 6.x)

---

## ESP32 Setup Steps

1. Open `esp32_medwatch/main.ino` in Arduino IDE
2. Update WiFi credentials:
   ```cpp
   const char* WIFI_SSID     = "YourNetworkName";
   const char* WIFI_PASSWORD = "YourPassword";
   ```
3. Find your phone's local IP address (Settings → WiFi → IP address)
4. Update phone IP:
   ```cpp
   const char* PHONE_IP = "192.168.1.XXX";
   ```
5. Select board: ESP32 Dev Module
6. Upload

---

## Phone App Setup Steps

### 1. Install Python dependencies
```bash
pip install flask numpy scipy scikit-learn joblib pandas tqdm
```

### 2. Download datasets
**BP dataset (UCI):**
- Download from jeya-maria-jose repo Google Drive link
- Place CSV at: `phone_app/data/uci/features.csv`

**Glucose dataset:**
```bash
pip install kaggle
kaggle datasets download muhammadyasirsaleem/ppg-signal-with-blood-sugar-level-data
unzip *.zip -d phone_app/data/glucose/
```

### 3. Train models (run once)
```bash
cd phone_app
python train_models.py
```

### 4. Start the server
```bash
python server.py
```

Server starts at `http://0.0.0.0:5000`
Make sure your phone and ESP32 are on the same WiFi network.

---

## API Endpoints

### POST /predict
Receives a window from ESP32, returns vitals.

**Request:**
```json
{
  "ppg": [0.12, 0.15, 0.18, ...],
  "ecg": [-0.3, -0.2, 0.1, ...]
}
```
**Response:**
```json
{
  "SBP": 120.5,
  "DBP": 78.2,
  "glucose": 95.0,
  "HR": 72.0,
  "PTT_ms": 312.5,
  "calibrated": false,
  "quality": "good",
  "timestamp": 1710000000,
  "inference_ms": 45.2
}
```

### POST /calibrate
Provide cuff reference measurements for personal calibration.

**Request:**
```json
{
  "pred_sbp": [118, 122, 115],
  "true_sbp": [122, 126, 118],
  "pred_dbp": [76, 79, 74],
  "true_dbp": [78, 82, 77]
}
```

### GET /history?n=20
Returns last 20 predictions (for graphing in your app UI).

### GET /status
Health check — confirms server is running and models are loaded.

---

## Motion Threshold Tuning

The MPU6050 motion threshold in `main.ino` is set to `0.12g` by default.
To tune it for your specific setup:

1. Upload the firmware
2. Open Serial Monitor (115200 baud)
3. Keep watch still — note the motion values printed
4. Walk around — note the motion values
5. Set `MOTION_THRESHOLD` to a value between the two

---

## Expected Accuracy

With the UCI dataset and Random Forest model, typical results are:
- **SBP:** MAE 4-7 mmHg, STD 6-9 mmHg
- **DBP:** MAE 3-5 mmHg, STD 5-7 mmHg
- **AAMI standard:** MAE ≤ 5 mmHg, STD ≤ 8 mmHg

After per-user calibration (3-5 cuff measurements), errors typically
reduce by 30-50%.
