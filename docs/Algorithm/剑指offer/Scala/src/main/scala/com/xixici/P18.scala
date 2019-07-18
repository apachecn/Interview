package com.xixici

import com.xixici.util.TreeNode

/**
  * Created by xixici
  * Date: 2019/3/14 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P18 {
  def Mirror(tree: TreeNode): TreeNode = {
    if (tree == null) return null
    val tmp = tree.left
    tree.left = tree.right
    tree.right = tmp
    Mirror(tree.left)
    Mirror(tree.right)
    tree
  }
}
