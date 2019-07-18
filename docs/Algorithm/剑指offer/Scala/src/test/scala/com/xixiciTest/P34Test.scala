package com.xixiciTest

import com.xixici.P34
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/22 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P34Test extends FunSuite {
  test("P34Test") {
    val Expected = 4
    val Actual = P34.FirstNotRepeatingChar("AABBeCCDD")
    assert(Actual === Expected)
  }
}
