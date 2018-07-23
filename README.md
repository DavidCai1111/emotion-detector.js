# emotion-detector.js
[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg)](http://standardjs.com/)
[![Build Status](https://travis-ci.org/DavidCai1993/emotion-detector.js.svg?branch=master)](https://travis-ci.org/DavidCai1993/emotion-detector.js)

Emotion recognition in Node.js, using [TensorFlow.js](https://js.tensorflow.org/) and [OpenCV](https://github.com/justadudewhohacks/opencv4nodejs).

## Some Examples

![faces-result.jpg](http://dn-cnode.qbox.me/FtE1eFwzKZJI8OhkvgIMD5eHaERx)
![ronaldo-result.jpg](http://dn-cnode.qbox.me/FhndmCXXRbDPQg6z_ONA-P6zta0E)
![angry-happy.jpg](http://dn-cnode.qbox.me/FoZG8X6nI66-ER9eJhp0qEPXJHYE)
## Installation

```sh
npm i emotion-detector -g
```

## How To Use It

### Draw The Result On Given Image (Like `Examples`)

```js
emotion-detector-js draw -i ./images/input.jpg -o ./output.jpg -c white
```

### Output The Result To Terminal

```js
emotion-detector-js info -i ./images/input.jpg
// Output:
// [ { face: { x: 251, y: 58, height: 82, width: 82 },
//     emotion: 'happy' },
//   { face: { x: 478, y: 73, height: 88, width: 88 },
//     emotion: 'happy' },
//   { face: { x: 24, y: 80, height: 100, width: 100 },
//     emotion: 'happy' },
//   { face: { x: 112, y: 97, height: 87, width: 87 },
//     emotion: 'happy' } ]
```
