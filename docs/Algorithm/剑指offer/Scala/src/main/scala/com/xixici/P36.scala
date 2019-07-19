package com.xixici

import com.xixici.util.ListNode

/**
  * Created by xixici
  * Date: 2019/3/22 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P36 {
  def FindFirstCommonNode(pHead1: ListNode, pHead2: ListNode): ListNode = {
    var p1 = pHead1
    var p2 = pHead2
    while (p1 != p2) {
      p1 = p1.next
      p2 = p2.next
      if (p1 != p2) {
        if (p1 == null) p1 = pHead1
        if (p2 == null) p2 = pHead2
      }
    }
    p1
  }
}
