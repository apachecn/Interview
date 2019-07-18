package com.xixiciTest

import com.xixici.P62
import com.xixici.util.TreeNode
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/5/9 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P62Test extends FunSuite {
  val a = new TreeNode(2)
  val b = new TreeNode(3)
  val c = new TreeNode(4)
  val d = new TreeNode(5)
  val e = new TreeNode(6)
  val f = new TreeNode(7)
  val g = new TreeNode(8)
  d.left = b
  d.right = f
  b.left = a
  b.right = c
  f.left = e
  f.right = g
  test("P62Test") {
    val Expected = 3
    val Actual = P62.KthNode(d, 2)
    assert(Actual.data === Expected)
  }
}
