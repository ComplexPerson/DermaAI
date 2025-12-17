const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  predictImage: (imagePath) => ipcRenderer.invoke('predict-image', imagePath)
})
