package com.xixiciTest

import com.xixici.P53
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/4/29 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P53Test extends FunSuite {
  test("P53Test") {
    val Expected = true
    val Actual = P53.isNumeric("+100".toCharArray)
    assert(Actual === Expected)
  }
  test("P53Test-2") {
    val Expected = false
    val Actual = P53.isNumeric("12e".toCharArray)
    assert(Actual === Expected)
  }

}