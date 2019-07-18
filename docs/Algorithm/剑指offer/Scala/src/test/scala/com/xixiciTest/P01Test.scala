package com.xixiciTest

import com.xixici.P01
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/5 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/

class P01Test extends FunSuite {

  var a: Array[Array[Int]] = Array(Array(1, 2, 8, 9), Array(2, 4, 9, 12), Array(4, 7, 10, 13), Array(6, 8, 11, 15))

  test("P1") {
    assert(P01.find(-1, a) === false)
    assert(P01.find(7, a) === true)
  }
}
