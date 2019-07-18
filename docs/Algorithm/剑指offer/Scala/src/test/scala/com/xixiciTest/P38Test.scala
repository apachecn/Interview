package com.xixiciTest

import com.xixici.P38
import com.xixici.util.TreeNode
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/25 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P38Test extends FunSuite {

  val node1 = new TreeNode(7)

  root1 = new TreeNode(8)
  val node2 = new TreeNode(2)
  val node3 = new TreeNode(9)
  val node4 = new TreeNode(2)
  val node5 = new TreeNode(4)
  val node6 = new TreeNode(7)
  val a = new TreeNode(9)
  val b = new TreeNode(2)
  var root1: TreeNode = _
  root1.left = node1
  root1.right = node2
  node1.left = node3
  node1.right = node4
  node4.left = node5
  node4.right = node6
  test("P38Test") {
    val Expected = 4
    val Actual = P38.TreeDepth(root1)
    assert(Actual === Expected)
  }
}