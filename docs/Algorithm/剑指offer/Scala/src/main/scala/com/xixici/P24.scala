package com.xixici

import com.xixici.util.TreeNode

/**
  * Created by xixici
  * Date: 2019/3/16 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P24 {
  val listAll = new collection.mutable.ListBuffer[collection.immutable.List[Int]]()
  val list = new collection.mutable.ListBuffer[Int]()

  def findPath(root: TreeNode, targetO: Int): List[List[Int]] = {
    var target = targetO
    if (root == null) return listAll.toList
    list.append(root.data)
    target -= root.data
    if (target == 0 && root.left == null & root.right == null)
      listAll.append(list.toList)
    findPath(root.left, target)
    findPath(root.right, target)
    list.remove(list.size - 1)
    listAll.toList
  }
}
