package com.xixiciTest

import com.xixici.P32
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/21 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P32Test extends FunSuite {
  val A = Array(3, 32, 321)
  test("P32Test") {
    val Expected = 321323.toString
    val Actual = P32.PrintMinNumber(A)
    assert(Actual === Expected)
  }
}