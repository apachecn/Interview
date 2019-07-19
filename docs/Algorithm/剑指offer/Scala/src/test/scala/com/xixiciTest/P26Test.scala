package com.xixiciTest

import com.xixici.P26
import com.xixici.util.TreeNode
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/19 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P26Test extends FunSuite {
  val node1 = new TreeNode(6)
  root = new TreeNode(10)
  val node2 = new TreeNode(14)
  val node3 = new TreeNode(4)
  val node4 = new TreeNode(8)
  val node5 = new TreeNode(12)
  val node6 = new TreeNode(16)
  var root: TreeNode = _
  root.left = node1
  root.right = node2
  node1.left = node3
  node1.right = node4
  node2.left = node5
  node2.right = node6
  test("P26Test") {

    val Expected = List(4, 6, 8, 10, 12, 14, 16)
    var res = P26.Convert(root)
    val Actual = collection.mutable.ListBuffer[Int]()
    while (res != null) {
      Actual.append(res.data)
      res = res.right
    }
    assert(Actual.toList === Expected)
  }

}
