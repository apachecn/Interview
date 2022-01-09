package com.xixiciTest

import com.xixici.P44
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/4/26 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P44Test extends FunSuite {
  test("P44Test") {
    val Expected = "student a am I"
    val Actual = P44.ReverseSentence("I am a student")
    assert(Actual === Expected)
  }
}

