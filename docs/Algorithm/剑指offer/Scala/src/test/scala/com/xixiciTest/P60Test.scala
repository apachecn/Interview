package com.xixiciTest

import com.xixici.P60
import com.xixici.util.TreeNode
import org.scalatest.FunSuite

import scala.collection.mutable.ArrayBuffer

/**
  * Created by xixici
  * Date: 2019/5/9 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P60Test extends FunSuite {
  val a = new TreeNode(1)
  val b = new TreeNode(2)
  val bb = new TreeNode(3)
  val c = new TreeNode(4)
  val d = new TreeNode(5)
  a.left = b
  b.left = c
  b.right = d
  a.right = bb
  bb.left = d
  bb.right = c


  test("P60Test") {
    val Expected = ArrayBuffer(ArrayBuffer(1), ArrayBuffer(2, 3), ArrayBuffer(4, 5, 5, 4))
    val Actual = P60.Print(a)
    assert(Actual === Expected)
  }
}