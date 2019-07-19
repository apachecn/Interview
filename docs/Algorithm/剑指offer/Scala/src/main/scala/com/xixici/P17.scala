package com.xixici

import com.xixici.util.TreeNode

/**
  * Created by xixici
  * Date: 2019/3/14 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P17 {
  def HasSubtree(tree1: TreeNode, tree2: TreeNode): Boolean = {
    var flag: Boolean = false
    if (tree1 != null && tree2 != null) {
      flag = DoesTree1HaveTree2(tree1, tree2)
      if (!flag) {
        flag = HasSubtree(tree1.left, tree2) || HasSubtree(tree1.right, tree2)
      }
    }
    flag
  }

  def DoesTree1HaveTree2(tree1: TreeNode, tree2: TreeNode): Boolean = {
    if (tree1 == null && tree2 != null) return false
    if (tree2 == null) return true
    if (tree1.data != tree2.data) return false
    else {
      return DoesTree1HaveTree2(tree1.left, tree2.left) && DoesTree1HaveTree2(tree1.right, tree2.right)
    }
  }
}
