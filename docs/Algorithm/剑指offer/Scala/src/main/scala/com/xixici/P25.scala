package com.xixici

import com.xixici.util.RandomListNode

/**
  * Created by xixici
  * Date: 2019/3/19 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P25 {
  def Clone(pHead: RandomListNode): RandomListNode = {

    if (pHead == null) return null

    val newNode = new RandomListNode(pHead.data)

    newNode.random = pHead.random
    newNode.next = Clone(pHead.next)

    newNode

  }
}
