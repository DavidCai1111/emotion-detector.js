'use strict'
require('@tensorflow/tfjs-node')
const detector = require('commander')
const tf = require('@tensorflow/tfjs')
const cv = require('opencv4nodejs')
const pkg = require('./package')
const imageUtil = require('./lib/image')
const faceUtil = require('./lib/face')

const EMOTION_MODEL_PATH = `file://${__dirname}/models/fer2013/model.json`

detector.version(pkg.version)

detector
  .command('draw')
  .description('Detect faces and emotion in given image, and draw the result on it')
  .option('-i, --inputImagePath <path>', 'Path to the input image')
  .option('-o, --outputImagePath <path>', 'Path to the output image')
  .option('-c, --color <color>', 'Color of the emotion text and rectangles on faces')
  .action(function (opts) {
    ;(async function () {
      let { inputImagePath, outputImagePath, color } = opts
      if (!color) color = 'black'

      const colorVec = imageUtil.getColorVecByString(color)

      const emotionModel = await tf.loadModel(EMOTION_MODEL_PATH)

      const inputShape = [
        emotionModel.feedInputShapes[0][1],
        emotionModel.feedInputShapes[0][2]
      ]

      let imageRGB = await imageUtil.loadImage(inputImagePath, false)
      let imageGray = await imageUtil.loadImage(inputImagePath, true)

      const faces = await faceUtil.getFaces(imageGray)

      for (const face of faces) {
        const x = cv.Point2(face.x, face.y)
        const y = cv.Point2(face.x + face.width, face.y + face.height)
        imageRGB.drawRectangle(x, y, colorVec)

        let faceImage = await imageRGB.getRegion(face)
        let tensor = await faceUtil.preprocessToTensor(faceImage, inputShape)

        imageRGB.putText(await faceUtil.inferEmotion(tensor, emotionModel), x, 0, 1, colorVec)
      }

      await cv.imwriteAsync(outputImagePath, imageRGB)
    })(console.error)
  })

detector
  .command('info')
  .description('only output the infomation of faces position and emotion')
  .option('-i, --inputImagePath <path>', 'Path to the input image')
  .action(function (opts) {
    ;(async function () {
      let { inputImagePath } = opts

      const emotionModel = await tf.loadModel(EMOTION_MODEL_PATH)

      const inputShape = [
        emotionModel.feedInputShapes[0][1],
        emotionModel.feedInputShapes[0][2]
      ]

      let imageRGB = await imageUtil.loadImage(inputImagePath, false)
      let imageGray = await imageUtil.loadImage(inputImagePath, true)

      const faces = await faceUtil.getFaces(imageGray)

      const results = []

      for (const face of faces) {
        let faceImage = await imageRGB.getRegion(face)
        let tensor = await faceUtil.preprocessToTensor(faceImage, inputShape)

        results.push({
          face: { x: face.x, y: face.y, height: face.height, width: face.width },
          emotion: await faceUtil.inferEmotion(tensor, emotionModel)
        })
      }

      console.log(results)
    })(console.error)
  })

detector.parse(process.argv)
