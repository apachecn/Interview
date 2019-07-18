package com.xixiciTest

import com.xixici.P52
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/4/29 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P52Test extends FunSuite {
  test("P52Test") {
    val Expected = true
    val Actual = P52.matchPattern("aaa".toCharArray, "a.a".toCharArray)
    assert(Actual === Expected)
  }
  test("P52Test-2") {
    val Expected = false
    val Actual = P52.matchPattern("aaa".toCharArray, "a.b".toCharArray)
    assert(Actual === Expected)
  }

}
