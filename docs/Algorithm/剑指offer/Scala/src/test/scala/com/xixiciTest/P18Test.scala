package com.xixiciTest

import com.xixici.P18
import com.xixici.util.TreeNode
import org.scalatest.FunSuite

import scala.collection.mutable

/**
  * Created by xixici
  * Date: 2019/3/14 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P18Test extends FunSuite {
  val node1 = new TreeNode(7)
  val node2 = new TreeNode(2)
  val node3 = new TreeNode(9)
  val node4 = new TreeNode(2)
  val node5 = new TreeNode(4)
  val node6 = new TreeNode(7)
  var root1: TreeNode = _
  root1 = new TreeNode(8)
  root1.left = node1
  root1.right = node2
  node1.left = node3
  node1.right = node4
  node4.left = node5
  node4.right = node6
  var root2: TreeNode = _
  root2 = root1
  root2.left = node2
  root2.right = node1
  node1.left = node4
  node1.right = node3
  node4.left = node6
  node4.right = node5


  test("P18Test") {
    val Expected = P18Test.preOrder(root1)
    val Actual = P18Test.preOrder(P18.Mirror(root2))
    assert(Actual === Expected)
  }

}


object P18Test {
  // PreOrder Visit Btree
  private def preOrder(root: TreeNode): List[Int] = {
    if (root == null) return null
    val resultList = mutable.ListBuffer[Int]()
    val treeStack = mutable.ArrayStack[TreeNode]()
    treeStack.push(root)
    while (!treeStack.isEmpty) {
      val tempNode = treeStack.pop();
      if (tempNode != null) {
        resultList += tempNode.data
        treeStack.push(tempNode.right)
        treeStack.push(tempNode.left)
      }
    }
    resultList.toList
  }
}


