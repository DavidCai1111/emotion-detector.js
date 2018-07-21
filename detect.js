'use strict'
require('@tensorflow/tfjs-node')
const cv = require('opencv4nodejs')
const tf = require('@tensorflow/tfjs')
const imageUtil = require('./lib/image')
const faceUtil = require('./lib/face')

const EMOTION_MODEL_PATH = `file://${__dirname}/models/fer2013/model.json`

;(async function () {
  const emotionModel = await tf.loadModel(EMOTION_MODEL_PATH)

  const inputShape = [
    emotionModel.feedInputShapes[0][1],
    emotionModel.feedInputShapes[0][2]
  ]

  let imageRGB = await imageUtil.loadImage('./images/ronaldo.JPG', false)
  let imageGray = await imageUtil.loadImage('./images/ronaldo.JPG', true)

  const faces = await faceUtil.getFaces(imageGray)

  for (const face of faces) {
    const x = cv.Point2(face.x, face.y)
    const y = cv.Point2(face.x + face.width, face.y + face.height)
    imageRGB.drawRectangle(x, y, new cv.Vec3(255, 255, 255))

    let faceImage = await imageRGB.getRegion(face)
    let tensor = await faceUtil.preprocessToTensor(faceImage, inputShape)

    imageRGB.putText(await faceUtil.inferEmotion(tensor, emotionModel), x, 0, 1, new cv.Vec3(255, 255, 255))
  }

  await cv.imwriteAsync('./test.jpg', imageRGB)
})(console.error)
