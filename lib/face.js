'use strict'
const path = require('path')
const cv = require('opencv4nodejs')

const FACE_MODEL_PATH = path.join(__dirname, '../models/haarcascade_frontalface_default.xml')
const faceModel = new cv.CascadeClassifier(FACE_MODEL_PATH)

async function getFaces (image) {
  const faces = await faceModel.detectMultiScaleAsync(image)

  return faces
}

module.exports = { getFaces }
