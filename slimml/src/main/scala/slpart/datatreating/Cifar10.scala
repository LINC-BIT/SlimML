package slpart.datatreating

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.dataset.image.{BGRImgNormalizer, BGRImgToSample, BytesToBGRImg}
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import slpart.sltransformer.{LabeledPointToSample, SampleToLabeledPoint}

object Cifar10 {
  def trainSamples(path: String,sc: SparkContext,category: Int,zScore: Boolean = false,useComp: Boolean = false,
                   itqbitN: Int = 1,//设置的需要生成新属性的个数
                   itqitN: Int = 20, //压缩算法迭代次数
                   itqratioN: Int = 100, //压缩算法每itqratioN个属性中使用一个属性进行计算
                   upBound: Int = 20, //每个压缩点包含原始点个数上限
                   splitN: Double = 2.0,// svd产生itqBitN个新的属性,对每个新的属性划分成splitN份
                   isSparse: Boolean = false //输入数据是否为libsvm格式
                    ): RDD[Sample[Float]] = {
    val trainSample = if(zScore){
      System.out.println("zScore training Samples ...")
      val cifar10BR = sc.parallelize(Cifar10ByteRecord.loadTrain(path))
      val trans = BytesToBGRImg() ->
        BGRImgNormalizer(Cifar10ByteRecord.trainMean,Cifar10ByteRecord.trainStd) -> BGRImgToSample()
      if(useComp){
        System.out.println("generate compressed training Samples  ...\n +" +
          s"category: ${category} itqbitN: ${itqbitN} itqitN: ${itqitN} itqratioN: ${itqratioN} upBound: ${upBound}" +
          s" splitN: ${splitN} isSparse: ${isSparse}")
        CompDeal.singleLayerComp(category,SampleToLabeledPoint().apply(trans.apply(cifar10BR)),
          itqbitN = itqbitN,itqitN = itqitN,itqratioN = itqratioN,upBound = upBound,
          splitN = splitN,isSparse = isSparse)
      } else trans.apply(cifar10BR)
    }
    else{
      System.out.println("load LabeledPoints for training ...")
      val cifar10LP = sc.parallelize(Cifar10LabeledPoint.loadTrain(path))
      if(useComp){
        System.out.println("generate compressed training Samples  ...\n +" +
          s"category: ${category} itqbitN: ${itqbitN} itqitN: ${itqitN} itqratioN: ${itqratioN} upBound: ${upBound}" +
          s" splitN: ${splitN} isSparse: ${isSparse}")
        CompDeal.singleLayerComp(category,cifar10LP,
          itqbitN = itqbitN,itqitN = itqitN,itqratioN = itqratioN,upBound = upBound,
          splitN = splitN,isSparse = isSparse)
      } else LabeledPointToSample().apply(cifar10LP)
    }
    trainSample
  }
  def validationSamples(path: String,sc: SparkContext,zScore: Boolean = false): RDD[Sample[Float]] = {
    val validationSample = if(zScore){
      System.out.println("zScore validation Samples ...")
      val vBR = sc.parallelize(Cifar10ByteRecord.loadTest(path))
      val trans = BytesToBGRImg() ->
        BGRImgNormalizer(Cifar10ByteRecord.testMean,Cifar10ByteRecord.testStd) -> BGRImgToSample()
      trans.apply(vBR)
    }else{
      System.out.println("load LabeledPoints for validation ...")
      LabeledPointToSample().apply(sc.parallelize(Cifar10LabeledPoint.loadTest(path)))
    }
    validationSample
  }
}
