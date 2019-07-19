package com.xixiciTest

import com.xixici.P65
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/5/9 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P65Test extends FunSuite {
  val matrix = Array('a', 'b', 'c', 'e', 's', 'f', 'c', 's', 'a', 'd', 'e', 'e')
  val str1 = Array('b', 'c', 'c', 'e', 'd')
  val str2 = Array('a', 'b', 'c', 'b')
  println(P65.hasPath(matrix, 3, 4, str1))
  println(P65.hasPath(matrix, 3, 4, str2))
  test("P65Test") {
    val Expected = true
    val Actual = P65.hasPath(matrix, 3, 4, str1)
    assert(Actual === Expected)
  }
}
