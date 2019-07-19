package com.xixici

import com.xixici.util.TreeNode

/**
  * Created by xixici
  * Date: 2019/5/9 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P62 {
  var index = 0

  def KthNode(root: TreeNode, k: Int): TreeNode = {
    if (root != null) {
      var node = KthNode(root.left, k)
      if (node != null)
        return node
      index += 1
      if (index == k)
        return root
      node = KthNode(root.right, k)
      if (node != null)
        return node
    }
    null
  }
}
