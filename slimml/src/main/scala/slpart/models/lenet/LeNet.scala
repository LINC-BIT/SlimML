package slpart.models.lenet

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat

object LeNet {
  def apply(classNum: Int = 10): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5).setName("conv_1")) //(28-5+2*0)/1 +1 = 24    24x24x6
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2).setName("pool_1"))//(24-2+2*0)/2 + 1 = 12     12x12x6
      .add(SpatialConvolution(6, 12, 5, 5).setName("conv_2"))//(12-5+2*0)/1  +1 = 8    8x8x12
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2).setName("pool_2")) //(8-2+2*0)/2 +1 = 4     4x4x12
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100).setName("fc_3"))
      .add(Tanh())
      .add(Linear(100, classNum).setName("fc_4"))
      .add(LogSoftMax())
    model
  }
}

object Cifar10LeNet {
  def apply(classNum: Int = 10): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Reshape(Array(3, 32, 32)))
      .add(SpatialConvolution(3, 6, 5, 5).setName("conv_1")) //(32-5+2*0)/1 +1 = 28
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2).setName("pool_2"))//(28-2+2*0)/2 + 1 = 14
      .add(SpatialConvolution(6, 16, 5, 5).setName("conv_3"))//(14-5+2*0)/1  +1 = 10
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2).setName("pool_4")) //(10-2+2*0)/2 +1 = 5
      .add(Reshape(Array(16 * 5 * 5)))
      .add(Linear(16 * 5 * 5, 120).setName("fc_5"))
      .add(Tanh())
      .add(Linear(120,84).setName("fc_6"))
      .add(Tanh())
      .add(Linear(84, classNum).setName("output"))
      .add(LogSoftMax())
    model
  }
}


object ReLULeNet {
  def apply(classNum: Int = 10): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5).setName("conv_1")) //(28-5+2*0)/1 +1 = 24    24x24x6
      .add(ReLU(true))
      .add(SpatialMaxPooling(2, 2, 2, 2).setName("pool_1"))//(24-2+2*0)/2 + 1 = 12     12x12x6
      .add(SpatialConvolution(6, 12, 5, 5).setName("conv_2"))//(12-5+2*0)/1  +1 = 8    8x8x12
      .add(ReLU(true))
      .add(SpatialMaxPooling(2, 2, 2, 2).setName("pool_2")) //(8-2+2*0)/2 +1 = 4     4x4x12
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100).setName("fc_3"))
      .add(ReLU(true))
      .add(Linear(100, classNum).setName("fc_4"))
      .add(LogSoftMax())
    model
  }
}

object BNLeNet {
  def apply(classNum: Int = 10): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5).setName("conv_1")) //(28-5+2*0)/1 +1 = 24    24x24x6
      .add(SpatialBatchNormalization(6))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2).setName("pool_1"))//(24-2+2*0)/2 + 1 = 12     12x12x6
      .add(SpatialConvolution(6, 12, 5, 5).setName("conv_2"))//(12-5+2*0)/1  +1 = 8    8x8x12
      .add(SpatialBatchNormalization(12))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2).setName("pool_2")) //(8-2+2*0)/2 +1 = 4     4x4x12
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100).setName("fc_3"))
      .add(Tanh())
      .add(Linear(100, classNum).setName("fc_4"))
      .add(LogSoftMax())
    model
  }
}