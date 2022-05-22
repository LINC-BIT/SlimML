package slpart.models.lenet

import java.text.SimpleDateFormat
import java.util.Date

import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import slpart.datatreating.Mnist
import com.intel.analytics.bigdl.numeric.NumericFloat

object TrainMnist {
  val logger = Logger.getLogger("org")
  logger.setLevel(Level.OFF)

  import Utils._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args,new TrainParams()).map(param => {
      val curTime = new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date)
      val logFile = s"${curTime}-${param.appName}-bigdl.log"
      LoggerFilter.redirectSparkInfoLogs(logFile)
      val conf = Engine.createSparkConf()
        .setAppName(param.appName)
      val sc = new SparkContext(conf)
      Engine.init

      val start = System.nanoTime()
      val trainSamples = Mnist.trainSamples(param.folder,sc,param.classes,
        param.zScore,param.useComp,param.itqbitN,param.itqitN,param.itqratioN,
        param.upBound,param.splitN,param.isSparse)
      val validationSamples = Mnist.validationSamples(param.folder,sc,param.zScore)
      System.out.println(s"generate Aggregate Samples: ${(System.nanoTime()-start) * 1.0 / 1e9} seconds")

      val model = if(param.loadSnapshot && param.modelSnapshot.isDefined){
        Module.load[Float](param.modelSnapshot.get)
      }
      else{
        LeNet(param.classes)
      }
      val optimMethod = if(param.loadSnapshot && param.stateSnapshot.isDefined){
        OptimMethod.load[Float](param.stateSnapshot.get)
      }
      else{
        param.optMethod match {
          case "adam" => new Adam[Float](learningRate = param.learningRate,learningRateDecay = param.learningRateDecay)
          case "adadelta" => new Adadelta[Float]()
          case "rmsprop" => new RMSprop[Float](param.learningRate,param.learningRateDecay)
          case "ftrl" => new Ftrl[Float](param.learningRate)
          case _ => new SGD[Float](learningRate = param.learningRate,learningRateDecay = param.learningRateDecay,
            weightDecay = param.weightDecay,momentum = param.momentum,
            dampening = param.dampening,nesterov = param.nesterov)
        }
      }

      if(param.storeInitModel && param.storeInitModelPath.isDefined){
        System.out.println(s"save initial model in ${param.storeInitModelPath.get}")
        model.save(param.storeInitModelPath.get,true)
      }


      val optimizer = Optimizer(
        model = model,
        sampleRDD = trainSamples,
        criterion = ClassNLLCriterion[Float](),
        batchSize = param.batchSize
      )

      if(param.checkpoint.isDefined){
        optimizer.setCheckpoint(param.checkpoint.get,Trigger.severalIteration(param.checkpointIteration))
        if(param.overwriteCheckpoint) optimizer.overWriteCheckpoint()
      }

      val prestate = T(("taskDel",param.taskDel),
        ("taskStrategy",param.taskStrategy.trim.toLowerCase()),
        ("taskRatio",param.taskRatio),
        ("taskDrop",param.taskDrop),

        ("epochDel",param.epochDel),
        ("epochStrategy",param.epochStrategy.trim().toLowerCase()),
        ("epochRatio",param.epochRatio),
        ("epochDrop",param.epochDrop),

        ("notIteration",param.notIteration),
        ("notEpoch",param.notEpoch),
        ("layerName","conv_1"),
        ("gradName","gradWeight"),

        ("useComp",param.useComp),
        ("onlyComp",param.onlyComp),
        ("getGradient",param.getGradient)
      )
      // set user defined state
      optimizer.setState(prestate)

      optimizer.setValidation(
        trigger = Trigger.everyEpoch,
        sampleRDD = validationSamples,
        vMethods = Array(new Top1Accuracy[Float](),new Top5Accuracy[Float](),new Loss[Float]()),
        batchSize = param.batchSize
      )
      optimizer.setOptimMethod(optimMethod)
      optimizer.setEndWhen(Trigger.maxEpoch(param.maxEpoch))

      val trainedModel = optimizer.optimize()

      if(param.storeTrainedModel && param.storeTrainedModelPath.isDefined){
        System.out.println(s"save trained model in ${param.storeTrainedModelPath.get}")
        trainedModel.save(param.storeTrainedModelPath.get,overWrite = true)
        if(param.storeTrainedStatePath.isDefined) {
          System.out.println(s"save trained state in ${param.storeTrainedStatePath.get}")
          optimMethod.save(param.storeTrainedStatePath.get,overWrite = true)
        }
      }

      sc.stop()
    })
  }


}
