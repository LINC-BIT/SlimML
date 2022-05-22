package slpart.loaderTest

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Optimizer, Top1Accuracy, Trigger}
import org.apache.spark.rdd.RDD
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.{SparkConf, SparkContext}
import slpart.datatreating.{CompDeal, MnistLabeledPoint}
import slpart.sltransformer.LabeledPointToSample

object MnistTest {
  def main(args: Array[String]): Unit = {
    val logger = Logger.getLogger("org")
    logger.setLevel(Level.OFF)
    val conf = Engine.createSparkConf()
      .setAppName("MnistLoaderTest")
      .set("spark.task.maxFailures","1")
    val sc = new SparkContext(conf)
    Engine.init

    val mnistDir = "/home/hadoop/Documents/accML-Res-Store/mnist"
    val trainData = mnistDir + "/train-images-idx3-ubyte"
    val trainLabel = mnistDir + "/train-labels-idx1-ubyte"
    val validationData = mnistDir + "/t10k-images-idx3-ubyte"
    val validationLabel = mnistDir + "/t10k-labels-idx1-ubyte"

    val trainRdd = sc.parallelize(MnistLabeledPoint.loadTrain(trainData,trainLabel))
    val testRdd = sc.parallelize(MnistLabeledPoint.loadTest(validationData,validationLabel))

//    val trainSamples = CompDeal.singleLayerComp(10,trainRdd,3,20,100,10,2.0,false)
    val trainSamples = CompDeal.compTrain(10,trainRdd,1,20,100,10,2.0,false)

    val model = Sequential[Float]()
    model.add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5"))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5"))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100).setName("fc1"))
      .add(Tanh())
      .add(Linear(100, 10).setName("fc2"))
      .add(LogSoftMax())

    val optimizer = Optimizer(
      model = model,
      sampleRDD = trainSamples,
      criterion = ClassNLLCriterion[Float](),
      batchSize = 128
    )

    val prestate = T(("taskDel",false),
      ("taskStrategy","meandiv10"),
      ("taskRatio",5.0),
      ("taskDrop",0.0),

      ("epochDel",false),
      ("epochStrategy","meandiv10"),
      ("epochRatio",5.0),
      ("epochDrop",0.0),

      ("notIteration",2),
      ("layerName","conv1_5x5"),
      ("gradName","gradWeight"),

      ("useComp",true),
      ("getGradient",true)
    )
    // set user defined state
    optimizer.setState(prestate)

    val validationSamples = LabeledPointToSample().apply(testRdd)

    optimizer.setValidation(
      trigger = Trigger.everyEpoch,
      sampleRDD = validationSamples,
      vMethods = Array(new Top1Accuracy[Float]()),
      batchSize = 128
    )
    optimizer.setEndWhen(Trigger.maxEpoch(15))

    optimizer.optimize()


    System.out.println(s"trainRdd.count: ${trainRdd.count()}")
    System.out.println(s"testRdd.count: ${testRdd.count()}")
    System.out.println(s"trainSamples.count: ${trainSamples.count()}")

    sc.stop()
  }
}
