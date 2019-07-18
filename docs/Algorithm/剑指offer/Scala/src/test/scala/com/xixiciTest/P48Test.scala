package com.xixiciTest

import com.xixici.P48
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/16 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P48Test extends FunSuite {

  test("P47Test") {
    val Expected = 15
    val Actual = P48.Add(7, 8)
    assert(Actual === Expected)
  }

}
