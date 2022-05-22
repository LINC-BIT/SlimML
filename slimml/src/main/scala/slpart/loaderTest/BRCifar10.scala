package slpart.loaderTest

import java.text.SimpleDateFormat
import java.util.Date

import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.SGD.EpochStep
import com.intel.analytics.bigdl.optim._
import org.apache.spark.rdd.RDD
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.{SparkConf, SparkContext}
import slpart.datatreating.{Cifar10ByteRecord, CompDeal, MnistByteRecord}
import slpart.models.alexnet.AdaAlexnet
import slpart.sltransformer.{LabeledPointToSample, SampleToLabeledPoint}

object BRCifar10 {
  val logger = Logger.getLogger("org")
  logger.setLevel(Level.OFF)
  def main(args: Array[String]): Unit = {
    val curTime = new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date)
    val logFile = s"${curTime}-brcifar10-bigdl.log"
    LoggerFilter.redirectSparkInfoLogs(logFile)
    val conf = Engine.createSparkConf()
      .setAppName("BRCifar10LoaderTest")
      .set("spark.task.maxFailures","1")
    val sc = new SparkContext(conf)
    Engine.init

    val start = System.nanoTime()
    val cifar10Dir = "/home/hadoop/Documents/accML-Res-Store/cifar-10-batches-bin"

    val trainRdd = sc.parallelize(Cifar10ByteRecord.loadTrain(cifar10Dir))
    val testRdd = sc.parallelize(Cifar10ByteRecord.loadTest(cifar10Dir))

    val trans = BytesToBGRImg() -> BGRImgNormalizer(Cifar10ByteRecord.trainMean,Cifar10ByteRecord.trainStd)->
      BGRImgToSample() -> SampleToLabeledPoint()
    val xr = trans.apply(trainRdd)

//    val trainSamples = CompDeal.singleLayerComp(10,xr,1,20,100,5,2.0,false)

    val trainSamples = LabeledPointToSample().apply(xr)
    System.out.println(s"generate Aggregate Samples: ${(System.nanoTime()-start) * 1.0 / 1e9} seconds")
//    System.exit(0)

    val hasDropout: Boolean = false
    val model = AdaAlexnet(10,false)

    val optimizer = Optimizer(
      model = model,
      sampleRDD = trainSamples,
      criterion = ClassNLLCriterion[Float](),
      batchSize = 16
    )

    val prestate = T(("taskDel",true),
      ("taskStrategy","meandiv10"),
      ("taskRatio",5.0),
      ("taskDrop",0.0),

      ("epochDel",false),
      ("epochStrategy","meandiv10"),
      ("epochRatio",5.0),
      ("epochDrop",0.0),

      ("notIteration",2),
      ("layerName","conv_1"),
      ("gradName","gradWeight"),

      ("useComp",false),
      ("onlyComp",false),
      ("getGradient",false)
    )
    // set user defined state
    optimizer.setState(prestate)

    val validationSamples = (BytesToBGRImg()-> BGRImgNormalizer(Cifar10ByteRecord.testMean,Cifar10ByteRecord.testStd)->
      BGRImgToSample()).apply(testRdd)

    optimizer.setValidation(
      trigger = Trigger.everyEpoch,
      sampleRDD = validationSamples,
      vMethods = Array(new Top1Accuracy[Float]()),
      batchSize = 16
    )
    optimizer.setEndWhen(Trigger.maxEpoch(3))

//    val optimMethod = new SGD[Float](learningRate = 0.01,learningRateSchedule = new EpochStep(10,0.1))
    val optimMethod = new Adam[Float](0.001)
    optimizer.setOptimMethod(optimMethod)

//    for(curStep <- 1 to 10){
//      System.out.println(s"curStep: ${curStep}")
////      optimizer.setTrainData(trainSamples,16*(curStep % 4 + 1))
//      val epoch: Int = curStep * 1
//      //optimMethod.learningRate = 0.01
//      optimizer.setEndWhen(Trigger.maxEpoch(epoch))
//      optimizer.optimize()
//    }

    optimizer.optimize()

    sc.stop()
  }
}
