package com.xixiciTest

import com.xixici.P04
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/5 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P04Test extends FunSuite {


  var pre = Array(1, 2, 4, 7, 3, 5, 6, 8)
  var in = Array(4, 7, 2, 1, 5, 3, 8, 6)

  test("P04Test") {
//    assert(P04.reConstructBinaryTree(pre, in).data === 1)
        assert(P04.reConstructBinaryTree(pre, in).left.data === 2)
    //    assert(P04.reConstructBinaryTree(pre, in).right.data === 3)
  }
}
