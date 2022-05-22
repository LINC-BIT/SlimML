package slpart.models.resnet

import java.text.SimpleDateFormat
import java.util.Date

import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T, Table}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.SGD.EpochStep
import slpart.datatreating.Cifar10
import slpart.models.resnet.ResNet.ShortcutType

object ResNet20Step {
  val logger = Logger.getLogger("org")
  logger.setLevel(Level.OFF)

  def cifar10Decay(epoch: Int): Double =
    if(epoch >= 122) 2.0 else if(epoch >= 81) 1.0 else 0.0

  import Utils._
  def main(args: Array[String]): Unit = {
    trainParser.parse(args,new TrainParams()).map(param =>{
      val curTime = new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date)
      val logFile = s"${curTime}-${param.appName}-bigdl.log"
      LoggerFilter.redirectSparkInfoLogs(logFile)
      val conf = Engine.createSparkConf()
        .setAppName(param.appName)
      val sc = new SparkContext(conf)
      Engine.init

      val start = System.nanoTime()
      val trainSamples = Cifar10.trainSamples(param.folder,sc,param.classes,param.zScore,param.useComp,
        param.itqbitN,param.itqitN,param.itqratioN,param.upBound,param.splitN,param.isSparse)
      val validationSamples = Cifar10.validationSamples(param.folder,sc,param.zScore)
      System.out.println(s"generate Aggregate Samples: ${(System.nanoTime()-start) * 1.0 / 1e9} seconds")

      val shortcut: ShortcutType = param.shortcutType match {
        case "A" => ShortcutType.A
        case "B" => ShortcutType.B
        case  _ => ShortcutType.C
      }

      val model = if(param.loadSnapshot && param.modelSnapshot.isDefined){
        Module.load[Float](param.modelSnapshot.get)
      }
      else{
        val curModel = if(param.graphModel){
          ResNet.graph(param.classes,
            T("shortcutType" -> shortcut,"depth" -> param.depth,"optnet" -> param.optnet))
        }else{
          ResNet(param.classes,
            T("shortcutType" -> shortcut,"depth" -> param.depth,"optnet" -> param.optnet))
        }
        if(param.optnet){
          ResNet.shareGradInput(curModel)
        }
        ResNet.modelInit(curModel)
        curModel
      }

      val optimMethod = if(param.loadSnapshot && param.stateSnapshot.isDefined){
        OptimMethod.load[Float](param.stateSnapshot.get)
      }else{
        param.optMethod match {
          case "adam" => new Adam[Float](param.learningRate,param.learningRateDecay)
          case "adadelta" => new Adadelta[Float]()
          case "rmsprop" => new RMSprop[Float](learningRate = param.learningRate,learningRateDecay = param.learningRateDecay)
          case _ => new SGD[Float](learningRate = param.learningRate,learningRateDecay = param.learningRateDecay,
            weightDecay = param.weightDecay,momentum = param.momentum,dampening = param.dampening,
            nesterov = param.nesterov,learningRateSchedule = EpochStep(param.epochstep,param.epochgamma))
        }
      }

      if(param.storeInitModel && param.storeInitModelPath.isDefined){
        System.out.println(s"save initial model in ${param.storeInitModelPath.get}")
        model.save(param.storeInitModelPath.get,true)
      }


      val optimizer = Optimizer(
        model = model,
        sampleRDD = trainSamples,
        criterion = CrossEntropyCriterion[Float](),
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
        vMethods = Array(new Top1Accuracy[Float](),new Top5Accuracy[Float](),
          new Loss[Float](CrossEntropyCriterion[Float]())),
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
