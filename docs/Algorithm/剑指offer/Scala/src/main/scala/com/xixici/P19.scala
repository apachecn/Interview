package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/14 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P19 {
  def printMatrix(matrix: Array[Array[Int]]): Array[Int] = {
    val rows = matrix.length
    val cols = matrix(0).length
    val res = new collection.mutable.ArrayBuffer[Int]()
    if (rows == 0 || cols == 0) return null
    var left = 0
    var top = 0
    var bottom = rows - 1
    var right = cols - 1
    while (left <= right && top <= bottom) {
      for (i <- left until right) {
        res.append(matrix(top)(i))
      }
      for (i <- top until bottom) {
        res.append(matrix(i)(right))
      }
      if (top != bottom) {
        for (i <- right until left by -1) {
          res.append(matrix(bottom)(i))
        }
      }
      if (left != right) {
        for (i <- bottom until top by -1) {
          res.append(matrix(i)(left))
        }
      }
      left += 1
      top += 1
      bottom -= 1
      right -= 1
    }
    res.toArray
  }
}
