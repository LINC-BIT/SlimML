package com.intel.analytics.bigdl.optim

import org.apache.spark.AccumulatorParam

import scala.collection.mutable.ArrayBuffer


object AryBufAccumulator extends AccumulatorParam[ArrayBuffer[Double]]{
  def zero(initialValue: ArrayBuffer[Double]): ArrayBuffer[Double] = {
    ArrayBuffer[Double]()
  }
  def addInPlace(r1: ArrayBuffer[Double], r2: ArrayBuffer[Double]): ArrayBuffer[Double] = {
    r1 ++ r2
  }

}
