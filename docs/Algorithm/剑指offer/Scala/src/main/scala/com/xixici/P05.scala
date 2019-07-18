package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/10 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P05 {

  val stack1 = new scala.collection.mutable.Stack[Int]
  val stack2 = new scala.collection.mutable.Stack[Int]

  def push(a: Int) = {
    stack1.push(a)
  }

  def pop() = {
    if (stack2.isEmpty) {
      while (!stack1.isEmpty) {
        stack2.push(stack1.pop())
      }
    }
    if (stack2.isEmpty) {
      throw new Exception("delete error, queue is empty.")
    }
    stack2.pop()
  }

}
