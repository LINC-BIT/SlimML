package slpart.datatreating

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import com.intel.analytics.bigdl.numeric.NumericFloat
import slpart.sllsh.ZFHashLayer
import AccurateML.nonLinearRegression.ZFHash3
import slpart.slpartitioner.CategoryPartitioner

/**
  * 用于粗粒度压缩点的生成
  */
object CompDeal {
  val compScale: Double = 1000

  def singleLayerComp(category: Int,
                       data: RDD[LabeledPoint],
                       itqbitN: Int = 1,//设置的需要生成新属性的个数
                      itqitN: Int = 20, //压缩算法迭代次数
                      itqratioN: Int = 100, //压缩算法每itqratioN个属性中使用一个属性进行计算
                      upBound: Int = 20, //每个压缩点包含原始点个数上限
                      splitN: Double = 2.0,// svd产生itqBitN个新的属性,对每个新的属性划分成splitN份
                      isSparse: Boolean = false //输入数据是否为libsvm格式
                     ) = {
    val sc = data.sparkContext
    val hash = new ZFHashLayer(itqitN,itqratioN,upBound,isSparse, 1,sc, 0)
    val objectData = data.map(p => (p.label.toInt,p))
      .partitionBy(new CategoryPartitioner(category))
      .map(_._2)
      .mapPartitions(p => hash.zfHash(p.toIterable))

//    objectData.map(_._2.length).foreach(println)
    val maxOrigNum = objectData.map(_._2.length).max()
//    val minOrigNum = objectData.map(_._2.length).min()
    System.out.println(s"comp - orig 1 - ${maxOrigNum}")
    objectData.mapPartitions(p => new CustomAryIterator(p,maxOrigNum,Some(-1)))
  }


  def compTrain(category: Int,
                 data: RDD[LabeledPoint],
                itqbitN: Int = 1, //设置的需要生成新属性的个数
                itqitN: Int = 20, //压缩算法迭代次数
                itqratioN: Int = 100, //压缩算法每itqratioN个属性中使用一个属性进行计算
                upBound: Int = 20, //每个压缩点包含原始点个数上限
                splitN: Double = 2.0, // svd产生itqBitN个新的属性,对每个新的属性划分成splitN份
                isSparse: Boolean = false //输入数据是否为libsvm格式
               ) = {
    val oHash = new ZFHash3(itqbitN,itqitN,itqratioN,upBound,splitN,isSparse)
    val objectData = data.map(p => (p.label.toInt,p))
      .partitionBy(new CategoryPartitioner(category))
      .map(_._2)
      .mapPartitions(oHash.zfHashMap).map(_._1)

//    objectData.map(_.last.length).foreach(println)
    val maxOrigNum = objectData.map(_.last.length).max()
//    val minOrigNum = objectData.map(_.last.length).min()
    System.out.println(s"comp - orig 1 - ${maxOrigNum}")
    objectData.zipWithIndex().mapPartitions(p => new CustomArySampleIter(p,maxOrigNum,false,compScale,Some(-1)))
  }
}
