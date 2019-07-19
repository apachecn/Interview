package com.xixici

import com.xixici.util.TreeNode

/**
  * Created by xixici
  * Date: 2019/3/25 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P38 {
  def TreeDepth(root: TreeNode): Int = {
    if (root == null) {
      return 0
    }
    val left: Int = TreeDepth(root.left)
    val right: Int = TreeDepth(root.right)
    math.max(left, right) + 1
  }
}
