'use strict'
const jimp = require('jimp')
const tf = require('@tensorflow/tfjs')

async function loadImage (path, grayscale, targetSize) {
  const image = await jimp.read(path)

  if (grayscale) image.grayscale()
  if (targetSize) image.resize(targetSize.width, targetSize.height)

  const p = []

  image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
    p.push(this.bitmap.data[idx + 0])
    p.push(this.bitmap.data[idx + 1])
    p.push(this.bitmap.data[idx + 2])
  })

  return tf.tensor3d(p, [image.bitmap.width, image.bitmap.height, 3])
}

module.exports = { loadImage }
