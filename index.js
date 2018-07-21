'use strict'
require('@tensorflow/tfjs-node')
const cv = require('opencv4nodejs')
const imageUtil = require('./lib/image')
const faceUtil = require('./lib/face')

;(async function () {
  let imageRGB = await imageUtil.loadImage('./images/faces.jpg', false)
  let imageGray = await imageUtil.loadImage('./images/faces.jpg', true)

  const faces = await faceUtil.getFaces(imageGray)

  for (const face of faces.objects) {
    const x = cv.Point2(face.x, face.y)
    const y = cv.Point2(face.x + face.width, face.y + face.height)
    imageRGB.drawRectangle(x, y)
  }

  await cv.imwriteAsync('./test.jpg', imageRGB)
})(console.error)
