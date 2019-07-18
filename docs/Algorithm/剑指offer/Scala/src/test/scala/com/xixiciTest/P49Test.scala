package com.xixiciTest

import com.xixici.P49
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/17 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P49Test extends FunSuite {


  test("P47Test") {
    val Expected = 0
    val Actual = P49.StrToInt("abc")
    assert(Actual === Expected)
  }

}
