package slpart.models.zfnet

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn._

object ZFNet {
  def apply(classNum: Int,hasDropout: Boolean = false): Module[Float]= {
    val model = Sequential[Float]()
    model.add(Reshape(Array(3,224,224)))
    model.add(SpatialConvolution(3,96,7,7,2,2,1,1).setName("conv_1")) // (224-7+2*1)/2 + 1 = 110
      .add(SpatialBatchNormalization(96))
      .add(ReLU(true))
      .add(SpatialMaxPooling(3,3,2,2,1,1)) // (110-3+2*1)/2 + 1 = 55

    model.add(SpatialConvolution(96,256,5,5,2,2,0,0).setName("conv_2")) // (55-5+2*0)/2 + 1 = 26
      .add(SpatialBatchNormalization(256))
      .add(ReLU(true))
      .add(SpatialMaxPooling(3,3,2,2,1,1)) // (26-3+2*1)/2 + 1 = 13

    model.add(SpatialConvolution(256,384,3,3,1,1,1,1).setName("conv_3")) // (13-3+2*1)/1 + 1 = 13
      .add(SpatialBatchNormalization(384))
      .add(ReLU(true))

    model.add(SpatialConvolution(384,384,3,3,1,1,1,1).setName("conv_4")) // (13-3+2*1)/1 + 1 = 13
      .add(SpatialBatchNormalization(384))
      .add(ReLU(true))

    model.add(SpatialConvolution(384,256,3,3,1,1,1,1).setName("conv_5")) // (13-3+2*1)/1 + 1 = 13
      .add(SpatialBatchNormalization(256))
      .add(ReLU(true))
      .add(SpatialMaxPooling(3,3,2,2)) // (13 - 3+2*0)/2 + 1 = 6

    model.add(Reshape(Array(6*6*256))) //
    model.add(Linear(6*6*256,4096).setName("fc_6"))
      .add(ReLU(true))
    if(hasDropout) model.add(Dropout(0.5))

    model.add(Linear(4096,4096).setName("fc_7"))
      .add(ReLU(true))
    if(hasDropout) model.add(Dropout(0.5))

    model.add(Linear(4096,classNum).setName("fc_8"))
      .add(LogSoftMax())
    model
  }
}

object ZFNetForCifar10 {
  def apply(classNum: Int,hasDropout: Boolean = false, hasbn: Boolean = true): Module[Float]= {
    val model = Sequential[Float]()
    model.add(Reshape(Array(3,32,32)))
    model.add(SpatialConvolution(3,64,5,5,2,2,2,2).setName("conv_1")) // (32-5+2*2)/2 + 1 = 16
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
