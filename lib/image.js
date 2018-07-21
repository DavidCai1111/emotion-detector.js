'use strict'
const cv = require('opencv4nodejs')
const colorString = require('color-string')

async function loadImage (path, grayscale) {
  let image = await cv.imreadAsync(path)
  if (grayscale) image = await image.bgrToGrayAsync()

  return image
}

function getColorVecByString (str) {
  let rgb = colorString.get.rgb(str)

  if (!rgb) throw new Error(`Invalid color string: "${str}"`)

  if (rgb.length > 3) {
    rgb = rgb.slice(0, 3)
  }

  return new cv.Vec3(rgb[2], rgb[1], rgb[0])
}

module.exports = { loadImage, getColorVecByString }
