package slpart.models.alexnet

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat

object AlexNetForMNIST {
  def apply(classNum: Int,hasDropout: Boolean = false,hasbn: Boolean = true): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Reshape(Array(1, 28, 28)))
    model.add(SpatialConvolution(1,64,3,3,1,1).setName("conv_1"))// (28-3)/1 + 1 = 26 26x26x64
      if(hasbn) model.add(SpatialBatchNormalization(64))
    model.add(ReLU(true).setName("relu_1"))
    model.add(SpatialMaxPooling(3,3,2,2).setName("maxpool_1"))//(26-3)/2 + 1 = 12 12x12x64

    model.add(SpatialConvolution(64,128,3,3,1,1).setName("conv_2"))//(12-3)/1 + 1 = 10 10x10x128
      if(hasbn) model.add(SpatialBatchNormalization(128))
    model.add(ReLU(true).setName("relu_2"))
    model.add(SpatialMaxPooling(3,3,2,2).setName("maxpool_2"))//(10-3)/2 + 1 = 4 4x4x128

    model.add(SpatialConvolution(128,256,3,3,1,1,1,1).setName("conv_3"))//(4-3 + 2*1)/1 + 1 = 2 4x4x256
      if(hasbn) model.add(SpatialBatchNormalization(256))
    model.add(ReLU(true).setName("relu_3"))
    model.add(SpatialMaxPooling(2,2,1,1).setName("maxpool_3"))//(4-2)/1 + 1 = 3 3x3x256
    model.add(Reshape(Array(3*3*256)))

      model.add(Linear(3*3*256,1024).setName("fc_4"))
    model.add(ReLU(true).setName("relu_4"))

      if(hasDropout) model.add(Dropout(0.5).setName("dropout_4"))
    model.add(Linear(1024,1024).setName("fc_5"))
    model.add(ReLU(true).setName("relu_5"))

      if(hasDropout) model.add(Dropout(0.5).setName("dropout_5"))
    model.add(Linear(1024,classNum).setName("fc_6"))
    model.add(LogSoftMax())
    model
  }
}

object AlexNetForCIFAR10 {
  def apply(classNum: Int,hasDropout: Boolean = false,hasbn: Boolean = true): Module[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(3,32,32)))
    model.add(SpatialConvolution(3,64,3,3,1,1).setName("conv_1"))// (32-3)/1 + 1 = 30 30x30x64
    if(hasbn) model.add(SpatialBatchNormalization(64))
    model.add(ReLU(true).setName("relu_1"))
    model.add(SpatialMaxPooling(3,3,2,2).setName("maxpool_1"))//(30-3)/2 + 1 = 14 14x14x64

    model.add(SpatialConvolution(64,128,3,3,1,1).setName("conv_2"))//(14-3)/1 + 1 = 12 12x12x128
    if(hasbn) model.add(SpatialBatchNormalization(128))
    model.add(ReLU(true).setName("relu_2"))
    model.add(SpatialMaxPooling(3,3,2,2).setName("maxpool_2"))//(12-3)/2 + 1 = 5 5x5x128

    model.add(SpatialConvolution(128,256,3,3,1,1,1,1).setName("conv_3"))//(5-3 + 2*1)/1 + 1 = 5 5x5x256
    if(hasbn) model.add(SpatialBatchNormalization(256))
    model.add(ReLU(true).setName("relu_3"))
    model.add(SpatialMaxPooling(2,2,1,1).setName("maxpool_3"))//(5-2)/1 + 1 = 4 4x4x256
    model.add(Reshape(Array(4096)))

    model.add(Linear(4096,1024).setName("fc_4"))
    model.add(ReLU(true).setName("relu_4"))
    if(hasDropout) model .add(Dropout(0.5).setName("dropout_4"))
    model.add(Linear(1024,1024).setName("fc_5"))
    model.add(ReLU(true).setName("relu_5"))
    if(hasDropout) model .add(Dropout(0.5).setName("dropout_5"))
    model.add(Linear(1024,classNum).setName("fc_6"))
    model.add(LogSoftMax())
    model
  }
}

object  AlexNetStdCifar10{
  def apply(classNum: Int,hasDropout: Boolean = false,hasbn: Boolean = true): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Reshape(Array(3,32,32)))
    model.add(SpatialConvolution(3,64,3,3,2,2,1,1).setName("conv_1")) // (32-3+2*1)/2 + 1 = 16
    if(hasbn) model.add(SpatialBatchNormalization(64))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2,2,2,2).setName("mp_1")) // (16-2)/2 + 1 = 8

    model.add(SpatialConvolution(64,192,3,3,1,1,1,1).setName("conv_2"))  // (8-3+2*1)/1+1 = 8
    if(hasbn) model.add(SpatialBatchNormalization(192))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2,2,2,2).setName("mp_2")) // (8-2)/2 +1 = 4

    model.add(SpatialConvolution(192,384,3,3,1,1,1,1).setName("conv_3")) //(4-3+2*1)/1 +1 = 4
    if(hasbn) model.add(SpatialBatchNormalization(384))
    model.add(ReLU(true))

    model.add(SpatialConvolution(384,256,3,3,1,1,1,1).setName("conv_4")) // (4-3+2*1)/1+1 = 4
    if(hasbn) model.add(SpatialBatchNormalization(256))
    model.add(ReLU(true))

    model.add(SpatialConvolution(256,256,3,3,1,1,1,1).setName("conv_5")) // (4-3+2*1)/1+1 = 4
    if(hasbn) model.add(SpatialBatchNormalization(256))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2,2,2,2).setName("mp_3")) // (4-2)/2 + 1 = 2

    model.add(Reshape(Array(256*2*2)))

    if(hasDropout) model .add(Dropout(0.5).setName("dropout_4"))
    model.add(Linear(256*2*2,4096).setName("fc_6"))
    model.add(ReLU(true))
    if(hasDropout) model .add(Dropout(0.5).setName("dropout_4"))
    model.add(Linear(4096,4096).setName("fc_7"))
    model.add(ReLU(true))
    model.add(Linear(4096,classNum).setName("fc_8"))
    model.add(LogSoftMax())

    model
  }
}

object AdaAlexnet{
  def apply(classNum: Int,hasDropout: Boolean = false,hasbn: Boolean = true): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Reshape(Array(3,32,32)))
    model.add(SpatialConvolution(3,64,11,11,4,4,5,5).setName("conv_1"))
    if(hasbn) model.add(SpatialBatchNormalization(64))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2,2,2,2))

    model.add(SpatialConvolution(64,192,5,5,1,1,2,2))
    if(hasbn) model.add(SpatialBatchNormalization(192))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2,2,2,2))

    model.add(SpatialConvolution(192,384,3,3,1,1,1,1))
    if(hasbn) model.add(SpatialBatchNormalization(384))
    model.add(ReLU(true))

    model.add(SpatialConvolution(384,256,3,3,1,1,1,1))
    if(hasbn) model.add(SpatialBatchNormalization(256))
    model.add(ReLU(true))

    model.add(SpatialConvolution(256,256,3,3,1,1,1,1))
    if(hasbn) model.add(SpatialBatchNormalization(256))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2,2,2,2))

    model.add(Reshape(Array(256)))
      .add(Linear(256,256))
      .add(ReLU(true))
      .add(Linear(256,classNum))
      .add(LogSoftMax())
    model
  }
}