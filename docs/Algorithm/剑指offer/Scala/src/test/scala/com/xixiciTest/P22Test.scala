package com.xixiciTest

import com.xixici.P22
import com.xixici.util.TreeNode
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/15 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P22Test extends FunSuite {
  val root1 = new TreeNode(8)
  val node1 = new TreeNode(7)
  val node2 = new TreeNode(2)
  val node3 = new TreeNode(9)
  val node4 = new TreeNode(2)
  val node5 = new TreeNode(4)
  val node6 = new TreeNode(7)
  root1.left = node1
  root1.right = node2
  node1.left = node3
  node1.right = node4
  node4.left = node5
  node4.right = node6
  test("P22Test") {
    val Expected = List(root1.data, node1.data, node2.data, node3.data, node4.data, node5.data, node6.data)
    val Actual = P22.PrintFromTopToBottom(root1)
    assert(Actual === Expected)
  }
}
