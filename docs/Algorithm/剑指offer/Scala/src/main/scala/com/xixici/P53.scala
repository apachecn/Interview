package com.xixici

/**
  * Created by xixici
  * Date: 2019/4/27 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P53 {
  def isNumeric(str: Array[Char]): Boolean = {
    val strOri = str.mkString("")
    strOri.matches("[\\+-]?[0-9]*(\\.[0-9]*)?([eE][\\+-]?[0-9]+)?")


  }

}
