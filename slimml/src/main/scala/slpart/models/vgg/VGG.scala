package slpart.models.vgg

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat

object VggForCifar10{
  def apply(classNum: Int, hasDropout: Boolean = true,hasbn: Boolean = true): Module[Float] = {
    val vggBnDo = Sequential[Float]().add(Reshape(Array(3,32,32)))
    def convBNReLU(nInputPlane: Int, nOutPutPlane: Int)
    : Sequential[Float] = {
      vggBnDo.add(SpatialConvolution(nInputPlane, nOutPutPlane, 3, 3, 1, 1, 1, 1))
      if(hasbn) vggBnDo.add(SpatialBatchNormalization(nOutPutPlane, 1e-5))
      vggBnDo.add(ReLU(true))
      vggBnDo
    }

    vggBnDo.add(SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1).setName("conv_1"))
    if(hasbn) vggBnDo.add(SpatialBatchNormalization(64, 1e-5))
    vggBnDo.add(ReLU(true))

//    convBNReLU(3, 64)

    if (hasDropout) vggBnDo.add(Dropout((0.3)))
    convBNReLU(64, 64)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2))

    convBNReLU(64, 128)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(128, 128)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2))

    convBNReLU(128, 256)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(256, 256)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(256, 256)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2))

    convBNReLU(256, 512)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(512, 512)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(512, 512)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2))

    convBNReLU(512, 512)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(512, 512)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(512, 512)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2))
    vggBnDo.add(Reshape(Array(512)))

    val classifier = Sequential[Float]()
    if (hasDropout) classifier.add(Dropout(0.5))
    classifier.add(Linear(512, 512))
    //    classifier.add(BatchNormalization(512))
    classifier.add(ReLU(true))
    if (hasDropout) classifier.add(Dropout(0.5))
    classifier.add(Linear(512, classNum))
    classifier.add(LogSoftMax())
    vggBnDo.add(classifier)

    vggBnDo
  }
}

object Vgg_16 {
  def apply(classNum: Int, hasDropout: Boolean = true,hasbn: Boolean = false): Module[Float] = {
    val model = Sequential().add(Reshape(Array(3,32,32)))
    model.add(SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1).setName("conv_1"))
    if(hasbn) model.add(SpatialBatchNormalization(64))
    model.add(ReLU(true))
    model.add(SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(64))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(128))
    model.add(ReLU(true))
    model.add(SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(128))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(256))
    model.add(ReLU(true))
    model.add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(256))
    model.add(ReLU(true))
    model.add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(256))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(512))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(512))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(512))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(512))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(512))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(512))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(Reshape(Array(512)))
//    model.add(Linear(512 ,512))
//    model.add(ReLU(true))
//    model.add(Threshold(0, 1e-6))
    if (hasDropout) model.add(Dropout(0.5))
    model.add(Linear(512, classNum))
    model.add(LogSoftMax())

    model
  }
}

object Vgg_19 {
  def apply(classNum: Int, hasDropout: Boolean = true,hasbn: Boolean = false): Module[Float] = {
    val model = Sequential().add(Reshape(Array(3,32,32)))
    model.add(SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1).setName("conv_1"))
    if(hasbn) model.add(SpatialBatchNormalization(64))
    model.add(ReLU(true))
    model.add(SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(64))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(128))
    model.add(ReLU(true))
    model.add(SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(128))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(256))
    model.add(ReLU(true))
    model.add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(256))
    model.add(ReLU(true))
    model.add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(256))
    model.add(ReLU(true))
    model.add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(256))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(512))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(512))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(512))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(512))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(512))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(512))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(512))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    if(hasbn) model.add(SpatialBatchNormalization(512))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(Reshape(Array(512)))
//    model.add(Linear(512 ,512))
//    model.add(ReLU(true))
//    model.add(Threshold(0, 1e-6))
    if (hasDropout) model.add(Dropout(0.5))
    model.add(Linear(512, classNum))
    model.add(LogSoftMax())

    model
  }
}
