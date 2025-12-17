const fileInput = document.getElementById('fileInput')
const previewHolder = document.getElementById('previewHolder')
const predictBtn = document.getElementById('predictBtn')
const resultBox = document.getElementById('result')
let selectedFile = null

fileInput.addEventListener('change', (e) => {
  const f = e.target.files[0]
  if (!f) return
  selectedFile = f
  const imageURL = URL.createObjectURL(f)
  previewHolder.innerHTML = `<img src="${imageURL}" class="image-preview"/>`
  resultBox.textContent = 'Ready to predict'
})

predictBtn.addEventListener('click', async () => {
  if (!selectedFile) { resultBox.textContent = 'Select an image first'; return }
  resultBox.textContent = 'Running prediction...'
  const imageBuffer = await selectedFile.arrayBuffer()
  const res = await window.electronAPI.predictImage(imageBuffer)
  if (res.error) {
    resultBox.textContent = 'Error: ' + (res.error || JSON.stringify(res))
    return
  }
  const results = res.results || []
  let html = '<h3>Top predictions</h3>'
  html += '<ol>'
  for (const r of results) {
    html += `<li>${r.class} — ${(r.confidence*100).toFixed(2)}%</li>`
  }
  html += '</ol>'
  html += '<p style="font-size:0.9em;color:#666">Interpretation: This summary shows the top predicted classes and confidences. For medical use, consult a specialist.</p>'
  resultBox.innerHTML = html
})

window.onload = () => {
  const ppBox = document.getElementById('possible-predictions')
  const DISEASE_MAP = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
  }
  let html = '<h3>Possible Outcomes</h3>'
  html += '<ul>'
  for (const key in DISEASE_MAP) {
    html += `<li>${DISEASE_MAP[key]}</li>`
  }
  html += '</ul>'
  ppBox.innerHTML = html
}
