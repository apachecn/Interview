package com.xixici

import com.xixici.util.ListNode

/**
  * Created by xixici
  * Date: 2019/3/13 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P14 {
  def FindKthToTail(listNode: ListNode, k: Int): ListNode = {
    if (listNode == null || k <= 0)
      return null
    var p = listNode
    for (i <- 0 until k - 1) {
      if (p.next != null) p = p.next else return null
    }
    var q = listNode
    while (p.next != null) {
      p = p.next
      q = q.next
    }
    q
  }
}
