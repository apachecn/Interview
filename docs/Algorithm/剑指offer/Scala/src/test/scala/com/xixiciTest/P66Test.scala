package com.xixiciTest

import com.xixici.P66
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/5/9 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P66Test extends FunSuite {
  test("P66Test") {
    println(P66.movingCount(10, 10, 5))
    println(P66.movingCount(20, 20, 15))
    val Expected = 21
    val Actual = P66.movingCount(10, 10, 5)
    assert(Actual === Expected)
  }
}
