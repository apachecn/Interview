package com.xixiciTest

import com.xixici.P58
import com.xixici.util.TreeNode
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/5/8 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P58Test extends FunSuite {
  val a = new TreeNode(1)
  val b = new TreeNode(2)
  val bb = new TreeNode(2)
  val c = new TreeNode(3)
  val d = new TreeNode(4)
  a.left = b
  b.left = c
  b.right = d
  a.right = bb
  bb.left = d
  bb.right = c


  test("P58Test") {
    val Expected = true
    val Actual = P58.isSymmetrical(a)
    assert(Actual === Expected)
  }
}
