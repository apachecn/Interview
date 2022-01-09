package com.xixici

import com.xixici.util.TreeNode

import scala.collection.mutable

/**
  * Created by xixici
  * Date: 2019/3/15 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P22 {
  def PrintFromTopToBottom(root: TreeNode): List[Int] = {
    //    val list = collection.mutable.ListBuffer[Int]()
    //    val stack = new mutable.ArrayStack[TreeNode]()
    //    stack.push(root)
    //    list += stack.pop().data
    //    while(!stack.isEmpty){
    //      if (stack.top.left != null) stack.push(stack.top.left)
    //      if (stack.top.right != null) stack.push(stack.top.right)
    //    }

    if (root == null) return null
    val resultList = mutable.ListBuffer[Int]()
    val treeStack = mutable.ListBuffer[TreeNode]()
    treeStack.append(root)
    while (!treeStack.isEmpty) {
      val tempNode = treeStack.remove(0)
      resultList += tempNode.data
      if (tempNode.left != null) treeStack.append(tempNode.left)
      if (tempNode.right != null) treeStack.append(tempNode.right)

    }
    resultList.toList
  }
}
