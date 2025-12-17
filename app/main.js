const { app, BrowserWindow, ipcMain } = require('electron')
const path = require('path')
const ort = require('onnxruntime-node')
const fs = require('fs')
const jimp = require('jimp') // Assuming jimp is installed for image processing

const DISEASE_MAP = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
};

// Model and class names configuration
const MODEL_PATH = path.join(__dirname, '..', 'checkpoints', 'model.onnx')
const IMAGE_SIZE = 224 // Must match the size used during ONNX export
const METADATA_PATH = path.join(__dirname, '..', 'data', 'HAM10000_metadata.csv'); // Path to metadata CSV

let classNames = []; // To be loaded from metadata

async function loadClassNames() {
  try {
    const csv = fs.readFileSync(METADATA_PATH, 'utf8');
    const lines = csv.split('\n');
    if (lines.length > 1) {
      // Assuming 'dx' column exists and is used for class names
      const dxColumnIndex = lines[0].split(',').indexOf('dx');
      if (dxColumnIndex === -1) {
        throw new Error("CSV metadata must contain 'dx' column.");
      }
      const uniqueDx = new Set();
      for (let i = 1; i < lines.length; i++) {
        const parts = lines[i].split(',');
        if (parts.length > dxColumnIndex) {
          uniqueDx.add(parts[dxColumnIndex]);
        }
      }
      const abbreviatedClassNames = Array.from(uniqueDx).sort();
      classNames = abbreviatedClassNames.map(name => DISEASE_MAP[name] || name);
    }
  } catch (error) {
    console.error('Failed to load class names from metadata:', error);
    // Fallback or error handling for classNames
    classNames = ['lesion_type_1', 'lesion_type_2', 'lesion_type_3']; // Example fallback
  }
}

function createWindow () {
  const win = new BrowserWindow({
    width: 900,
    height: 700,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  })
  win.loadFile('index.html')
}

app.whenReady().then(async () => {
  await loadClassNames(); // Load class names before creating window
  createWindow()
  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit()
})


ipcMain.handle('predict-image', async (event, imageBuffer) => {
  if (!fs.existsSync(MODEL_PATH)) {
    return { error: `Model not found at ${MODEL_PATH}. Please train a model and export it to ONNX.` };
  }
  if (classNames.length === 0) {
    return { error: 'Class names not loaded. Check metadata CSV.' };
  }

  try {
    // 1. Preprocess the image from buffer
    const image = await jimp.read(imageBuffer);
    image.resize(IMAGE_SIZE, IMAGE_SIZE).quality(100);

    // Convert to a Float32Array (normalized to 0-1 and then mean/std)
    const [R, G, B] = [ [], [], [] ];
    image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
        R.push(image.bitmap.data[idx + 0] / 255.0);
        G.push(image.bitmap.data[idx + 1] / 255.0);
        B.push(image.bitmap.data[idx + 2] / 255.0);
    });

    const inputData = [];
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < R.length; i++) {
        inputData[i] = (R[i] - mean[0]) / std[0];
        inputData[i + R.length] = (G[i] - mean[1]) / std[1];
        inputData[i + 2 * R.length] = (B[i] - mean[2]) / std[2];
    }

    const inputTensor = new ort.Tensor('float32', Float32Array.from(inputData), [1, 3, IMAGE_SIZE, IMAGE_SIZE]);

    // 2. Load and run ONNX model
    const session = await ort.InferenceSession.create(MODEL_PATH);
    const feeds = { input: inputTensor };
    const results = await session.run(feeds);
    const output = results.output.data; // Assuming 'output' is the name of the output tensor

    // 3. Post-process (softmax and top-k)
    const probabilities = Array.from(output).map(Math.exp); // Apply exp for softmax-like behavior if log-softmax was used
    const sumProbabilities = probabilities.reduce((a, b) => a + b, 0);
    const normalizedProbabilities = probabilities.map(p => p / sumProbabilities);

    const predictions = normalizedProbabilities
      .map((prob, index) => ({ class: classNames[index], confidence: prob }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3); // Top 3 predictions

    return { results: predictions };

  } catch (error) {
    console.error('Prediction failed:', error);
    return { error: error.message };
  }
})