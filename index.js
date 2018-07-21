'use strict'
require('@tensorflow/tfjs-node')
const _ = require('lodash')
const cv = require('opencv4nodejs')
const tf = require('@tensorflow/tfjs')
const imageUtil = require('./lib/image')
const faceUtil = require('./lib/face')
const util = require('./lib/util')

const EMOTION_MODEL_PATH = `file://${__dirname}/models/fer2013/model.json`

;(async function () {
  const emotionModel = await tf.loadModel(EMOTION_MODEL_PATH)

  const inputShape = [
    emotionModel.feedInputShapes[0][1],
    emotionModel.feedInputShapes[0][2]
  ]

  let imageRGB = await imageUtil.loadImage('./images/faces.jpg', false)
  let imageGray = await imageUtil.loadImage('./images/faces.jpg', true)

  const faces = await faceUtil.getFaces(imageGray)

  for (const face of faces.objects) {
    const x = cv.Point2(face.x, face.y)
    const y = cv.Point2(face.x + face.width, face.y + face.height)
    imageRGB.drawRectangle(x, y)

    let faceImage = await imageRGB.getRegion(face)
    faceImage = await faceImage.resizeAsync(inputShape[0], inputShape[1])
    faceImage = await faceImage.bgrToGrayAsync()

    let tensor = tf.tensor3d(_.flattenDeep(faceImage.getDataAsArray()), [64, 64, 1])

    console.log(tensor.shape)
    tensor = tensor.asType('float32')
    tensor = tensor.div(255.0)
    tensor = tensor.sub(0.5)
    tensor = tensor.mul(2.0)
    tensor = tensor.reshape([1, 64, 64, 1])

    const result = await emotionModel.predict(tensor)
    imageRGB.putText(util.getEmotionLabel(result), x, 0, 1)
  }

  await cv.imwriteAsync('./test.jpg', imageRGB)
})(console.error)
