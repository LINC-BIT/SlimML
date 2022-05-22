package slpart.loaderTest

import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.apache.spark.rdd.RDD
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.{SparkConf, SparkContext}
import slpart.datatreating.{Cifar10LabeledPoint, CompDeal}



object Cifar10Test {
  def main(args: Array[String]): Unit = {
    val logger = Logger.getLogger("org")
    logger.setLevel(Level.OFF)
    val conf = Engine.createSparkConf()
      .setAppName("Cifar10LoaderTest")
      .set("spark.task.maxFailures","1")
    val sc = new SparkContext(conf)
    Engine.init

    val cifar10Dir = "/home/hadoop/Documents/accML-Res-Store/cifar-10-batches-bin"

    val trainRdd = sc.parallelize(Cifar10LabeledPoint.loadTrain(cifar10Dir)).cache()
    val testRdd = sc.parallelize(Cifar10LabeledPoint.loadTest(cifar10Dir)).cache()

    val trainSamples = CompDeal.compTrain(10,trainRdd,1,20,100,10,2.0,false)
    trainSamples.cache()


    System.out.println(s"trainRdd.count: ${trainRdd.count()}")
    System.out.println(s"testRdd.count: ${testRdd.count()}")
    System.out.println(s"trainSamples.count: ${trainSamples.count()}")

    sc.stop()
  }
}
