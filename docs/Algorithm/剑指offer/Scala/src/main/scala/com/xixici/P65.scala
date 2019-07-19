package com.xixici

/**
  * Created by xixici
  * Date: 2019/6/25 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P65 {

  def hasPath(matrix: Array[Char], rows: Int, cols: Int, str: Array[Char]): Boolean = {
    if (matrix == null || rows <= 0 || cols <= 0 || str == null) return false
    val visited = new Array[Boolean](rows * cols)
    val pathLen = Array(0)
    for (row <- 0 until rows) {
      for (col <- 0 until cols) {
        if (hasPathCore(matrix, str, rows, cols, row, col, visited, pathLen)) {
          return true
        }
      }
    }
    false
  }

  private def hasPathCore(matrix: Array[Char], str: Array[Char], rows: Int, cols: Int, row: Int, col: Int,
                          visited: Array[Boolean], pathLen: Array[Int]): Boolean = {
    if (pathLen(0) == str.length) return true

    var hasPath = false
    if (row >= 0 && row < rows && col >= 0 && col < cols && matrix(row * cols + col) == str(pathLen(0))
      && !visited(row * cols + col)) {
      pathLen(0) += 1
      visited(row * cols + col) = true
      hasPath = hasPathCore(matrix, str, rows, cols, row - 1, col, visited, pathLen) ||
        hasPathCore(matrix, str, rows, cols, row + 1, col, visited, pathLen) ||
        hasPathCore(matrix, str, rows, cols, row, col - 1, visited, pathLen) ||
        hasPathCore(matrix, str, rows, cols, row, col + 1, visited, pathLen)
      if (!hasPath) {
        pathLen(0) -= 1
        visited(row * cols + col) = false
      }
    }
    hasPath
  }
}
