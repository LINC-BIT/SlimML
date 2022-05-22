package slpart.datatreating

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import slpart.sltransformer.{LabeledPointToSample, SampleToLabeledPoint}

object Mnist {
  def trainSamples(path: String,sc: SparkContext,category: Int,zScore: Boolean = false,useComp: Boolean = false,
                   itqbitN: Int = 1,
                   itqitN: Int = 20, //压缩算法迭代次数
                   itqratioN: Int = 100, //压缩算法每itqratioN个属性中使用一个属性进行计算
                   upBound: Int = 20, //每个压缩点包含原始点个数上限
                   splitN: Double = 2.0,// svd产生itqBitN个新的属性,对每个新的属性划分成splitN份
                   isSparse: Boolean = false //输入数据是否为libsvm格式
                  ): RDD[Sample[Float]] = {
    val trainImagePath = path + "/train-images-idx3-ubyte"
    val trainLabelPath = path + "/train-labels-idx1-ubyte"
    val trainSample = if(zScore){
      System.out.println("zScore training Samples ...")
      val mnistBR = sc.parallelize(MnistByteRecord.loadTrain(trainImagePath,trainLabelPath))
      val trans = BytesToGreyImg(28,28) ->
        GreyImgNormalizer(MnistByteRecord.trainMean,MnistByteRecord.trainStd) -> GreyImgToSample()
      if(useComp){
        System.out.println("generate compressed training Samples  ...\n +" +
          s"category: ${category} itqbitN: ${itqbitN} itqitN: ${itqitN} itqratioN: ${itqratioN} upBound: ${upBound}" +
          s" splitN: ${splitN} isSparse: ${isSparse}")
        CompDeal.singleLayerComp(category,SampleToLabeledPoint().apply(trans.apply(mnistBR)),
          itqbitN = itqbitN, itqitN = itqitN, itqratioN = itqratioN, upBound = upBound,
          splitN = splitN, isSparse = isSparse)
      }else trans.apply(mnistBR)
    }
    else{
      System.out.println("load LabeledPoints for training ...")
      val mnistLP = sc.parallelize(MnistLabeledPoint.loadTrain(trainImagePath,trainLabelPath))
      if(useComp){
        System.out.println("generate compressed training Samples  ...\n +" +
          s"category: ${category} itqbitN: ${itqbitN} itqitN: ${itqitN} itqratioN: ${itqratioN} upBound: ${upBound}" +
          s" splitN: ${splitN} isSparse: ${isSparse}")
        CompDeal.singleLayerComp(category,mnistLP,
          itqbitN = itqbitN, itqitN = itqitN, itqratioN = itqratioN, upBound = upBound,
          splitN = splitN, isSparse = isSparse)
      }else LabeledPointToSample().apply(mnistLP)
    }
    trainSample
  }

  def validationSamples(path: String,sc: SparkContext,zScore: Boolean = false): RDD[Sample[Float]] = {
    val testImagePath = path + "/t10k-images-idx3-ubyte"
    val testLabelPath = path + "/t10k-labels-idx1-ubyte"
    val validationSample = if(zScore){
      System.out.println("zScore validation Samples ...")
      val vBR = sc.parallelize(MnistByteRecord.loadTest(testImagePath,testLabelPath))
      val trans = BytesToGreyImg(28,28) ->
      GreyImgNormalizer(MnistByteRecord.testMean,MnistByteRecord.testStd) -> GreyImgToSample()
      trans.apply(vBR)
    }else{
      System.out.println("load LabeledPoints for validation ...")
      LabeledPointToSample().apply(sc.parallelize(MnistLabeledPoint.loadTest(testImagePath,testLabelPath)))
    }
    validationSample
  }

}
