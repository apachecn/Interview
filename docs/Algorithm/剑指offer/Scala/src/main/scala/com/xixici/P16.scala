package com.xixici

import com.xixici.util.ListNode

/**
  * Created by xixici
  * Date: 2019/3/14 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P16 {
  def Merge(list1: ListNode, list2: ListNode): ListNode = {
    if (list1 == null) {
      return list2
    }
    if (list2 == null) {
      return list1
    }
    if (list1.data <= list2.data) {
      list1.next = Merge(list1.next, list2)
      return list1
    } else {
      list2.next = Merge(list1, list2.next)
      return list2
    }

  }
}
