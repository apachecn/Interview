package com.xixiciTest

import com.xixici.P27
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/20 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P27Test extends FunSuite {
  val A = "123"
  test("P28Test") {
    val Expected = List("123", "132", "213", "231", "321", "312")
    val Actual = P27.Permutation(A)
    assert(Actual === Expected)
  }
}
