'use strict'
const path = require('path')
const _ = require('lodash')
const tf = require('@tensorflow/tfjs')
const cv = require('opencv4nodejs')
const util = require('./util')

const FACE_MODEL_PATH = path.join(__dirname, '../models/haarcascade_frontalface_default.xml')
const faceModel = new cv.CascadeClassifier(FACE_MODEL_PATH)

async function getFaces (image) {
  const facesResult = await faceModel.detectMultiScaleAsync(image)

  const faces = facesResult.objects.map(function (face, i) {
    if (facesResult.numDetections[i] < 10) return null

    return face
  })

  return faces.filter(function (face) { return face })
}

async function preprocessToTensor (faceImage, targetSize) {
  faceImage = await faceImage.resizeAsync(targetSize[0], targetSize[1])
  faceImage = await faceImage.bgrToGrayAsync()

  let tensor = tf.tensor3d(_.flattenDeep(faceImage.getDataAsArray()), [64, 64, 1])

  tensor = tensor.asType('float32')
  tensor = tensor.div(255.0)
  tensor = tensor.sub(0.5)
  tensor = tensor.mul(2.0)
  tensor = tensor.reshape([1, 64, 64, 1])

  return tensor
}

async function inferEmotion (tensor, model) {
  const result = await model.predict(tensor)

  return util.getEmotionLabel(result)
}

module.exports = { getFaces, preprocessToTensor, inferEmotion }
