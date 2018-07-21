'use strict'
const cv = require('opencv4nodejs')

async function loadImage (path, grayscale) {
  let image = await cv.imreadAsync(path)
  if (grayscale) image = await image.bgrToGrayAsync()

  return image
}

module.exports = { loadImage }
