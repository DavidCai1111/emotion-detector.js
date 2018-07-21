'use strict'
require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const imageUtil = require('./lib/image')

;(async function () {
  const rgbImage = await imageUtil.loadImage('./images/faces.jpg', false)
  const grayImage = await imageUtil.loadImage('./images/faces.jpg', true)

  rgbImage.print()
  grayImage.print()

  await imageUtil.saveImage('./test1.jpg', grayImage)
  await imageUtil.saveImage('./test2.jpg', rgbImage)
})(console.error)
