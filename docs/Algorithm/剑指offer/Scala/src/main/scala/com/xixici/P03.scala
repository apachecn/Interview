package com.xixici


import com.xixici.util.ListNode

/**
  * Created by xixici
  * Date: 2019/3/5 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P03 {
  def printListFromTailToHead(head: ListNode): collection.immutable.List[ListNode] = {
    if (head == null) {

    }
    val list = new collection.mutable.ListBuffer[ListNode]()
    val stack = collection.mutable.ArrayStack[ListNode]()
    var _head = head
    while (_head != null) {
      stack.push(_head)
      _head = _head.next
    }
    while (!stack.isEmpty) {
      list.append(stack.pop())
    }
    list.toList
  }
}
