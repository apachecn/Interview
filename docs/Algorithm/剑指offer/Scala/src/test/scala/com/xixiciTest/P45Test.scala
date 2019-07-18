package com.xixiciTest

import com.xixici.P45
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/4/26 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P45Test extends FunSuite {
  test("P45Test") {
    val Expected = true
    val Actual = P45.isContinuous(Array(4, 5, 6, 0, 0))
    assert(Actual === Expected)
  }
}

