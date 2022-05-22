package slpart.models.lenet

import scopt.OptionParser

object Utils {
  case class TrainParams(
                          appName: String = "appName",
                          folder: String = "./",
                          checkpoint: Option[String] = None,
                          checkpointIteration: Int = 10000,
                          overwriteCheckpoint: Boolean = true,
                          loadSnapshot: Boolean = false,
                          modelSnapshot: Option[String] = None,
                          stateSnapshot: Option[String] = None,
                          optnet: Boolean = false,
                          depth: Int = 20,
                          classes: Int = 10,
                          shortcutType: String = "A",
                          batchSize: Int = 128,
                          maxEpoch: Int = 165,
                          maxIteration: Int = 100000,
                          learningRate: Double = 0.05,
                          learningRateDecay: Double = 0.0,
                          weightDecay: Double = 0.0,
                          momentum: Double = 0.0,
                          dampening: Double = Double.MaxValue,
                          nesterov: Boolean = false,
                          graphModel: Boolean = false,
                          warmupEpoch: Int = 0,
                          maxLr: Double = 0.0,
                          itqbitN: Int = 1, // custom options
                          itqitN: Int = 20,
                          itqratioN: Int = 100,
                          minPartN: Int = 1,
                          upBound: Int = 20,
                          splitN : Double = 2.0,
                          isSparse: Boolean = false,
                          taskDel: Boolean = false,
                          taskStrategy: String = "default",
                          taskRatio: Double = 0.01,
                          taskDrop: Double = 0.05,
                          epochDel: Boolean = false,
                          epochStrategy: String = "default",
                          epochRatio: Double = 0.001,
                          epochDrop: Double = 0.05,
                          notIteration: Int = 0,
                          optMethod: String = "sgd",
                          useComp: Boolean = false,
                          onlyComp: Boolean = false,
                          getGradient: Boolean = false,
                          storeInitModel: Boolean = false,
                          storeInitModelPath: Option[String] = None,
                          storeInitStatePath: Option[String] = None,
                          storeTrainedModel: Boolean = false,
                          storeTrainedModelPath: Option[String] = None,
                          storeTrainedStatePath: Option[String] = None,
                          zScore: Boolean = false,
                          lrScheduler: Option[String] = None,
                          hasDropout: Boolean = false,
                          notEpoch: Int = 5,
                          hasbn: Boolean = false)

  val trainParser = new OptionParser[TrainParams]("BigDL LeNet") {
    head("Train LeNet model")
    opt[String]("appName")
      .text("the application name")
      .action((x,c) => c.copy(appName = x))
    opt[String]('f', "folder")
      .text("where you put your training files")
      .action((x, c) => c.copy(folder = x))
    opt[Boolean]("loadSnapshot")
      .text("if load snapshot")
      .action((x,c) => c.copy(loadSnapshot = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("checkpoint")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[Int]("checkpointIteration")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpointIteration = x))
    opt[Boolean]("overwriteCheckpoint")
      .text("where to cache the model")
      .action((x, c) => c.copy(overwriteCheckpoint = x))
    opt[Boolean]("optnet")
      .text("shared gradients and caches to reduce memory usage")
      .action((x, c) => c.copy(optnet = x))
    opt[Int]("depth")
      .text("depth of ResNet, 18 | 20 | 34 | 50 | 101 | 152 | 200")
      .action((x, c) => c.copy(depth = x))
    opt[Int]("classes")
      .text("classes of ResNet")
      .action((x, c) => c.copy(classes = x))
    opt[String]("shortcutType")
      .text("shortcutType of ResNet, A | B | C")
      .action((x, c) => c.copy(shortcutType = x))
    opt[Int]("batchSize")
      .text("batchSize of ResNet, 64 | 128 | 256 | ..")
      .action((x, c) => c.copy(batchSize = x))
    opt[Int]("maxEpoch")
      .text("number of epochs of leNet; default is 165")
      .action((x, c) => c.copy(maxEpoch = x))
    opt[Int]("maxIteration")
      .text("number of iteration of leNet; default is 100000")
      .action((x,c) => c.copy(maxIteration = x))
    opt[Double]("learningRate")
      .text("initial learning rate of leNet; default is 0.1")
      .action((x, c) => c.copy(learningRate = x))
    opt[Double]("learningRateDecay")
      .text("initial learningDecay rate of LeNet; default is 0.0")
      .action((x, c) => c.copy(learningRateDecay = x))
    opt[Double]("momentum")
      .text("momentum of ResNet; default is 0.9")
      .action((x, c) => c.copy(momentum = x))
    opt[Double]("weightDecay")
      .text("weightDecay of ResNet; default is 1e-4")
      .action((x, c) => c.copy(weightDecay = x))
    opt[Double]("dampening")
      .text("dampening of ResNet; default is 0.0")
      .action((x, c) => c.copy(dampening = x))
    opt[Boolean]("nesterov")
      .text("nesterov of ResNet; default is trye")
      .action((x, c) => c.copy(nesterov = x))
    opt[Boolean]('g', "graphModel")
      .text("use graph model")
      .action((x, c) => c.copy(graphModel = x))
    opt[Int]("warmupEpoch")
      .text("warmup epoch")
      .action((x, c) => c.copy(warmupEpoch = x))
    opt[Double]("maxLr")
      .text("maxLr")
      .action((x, c) => c.copy(maxLr = x))

    opt[Int]("itqbitN")
      .text("itqbitN")
      .action((x,c) => c.copy(itqbitN = x))
    opt[Int]("itqitN")
      .text("itqitN")
      .action((x,c) => c.copy(itqitN = x))
    opt[Int]("itqrationN")
      .text("itqratioN")
      .action((x,c) => c.copy(itqratioN = x))
    opt[Int]("minPartN")
      .text("minPartN")
      .action((x,c) => c.copy(minPartN = x))
    opt[Int]("upBound")
      .text("upBound")
      .action((x,c) => c.copy(upBound = x))
    opt[Double]("splitN")
      .text("splitN")
      .action((x,c) => c.copy(splitN = x))
    opt[Boolean]("isSparse")
      .text("isSparse")
      .action((x, c) => c.copy(isSparse = x))

    opt[Boolean]("taskDel")
      .text("task level filter")
      .action((x,c) => c.copy(taskDel = x))
    opt[String]("taskStrategy")
      .text("task level del strategy")
      .action((x,c) => c.copy(taskStrategy = x))
    opt[Double]("taskRatio")
      .text("task delete ratio")
      .action((x,c) => c.copy(taskRatio = x))
    opt[Double]("taskDrop")
      .text("task delete drop decay")
      .action((x,c) => c.copy(taskDrop = x))

    opt[Boolean]("epochDel")
      .text("epoch delete")
      .action((x,c) => c.copy(epochDel = x))
    opt[String]("epochStrategy")
      .text("epoch level del strategy")
      .action((x,c) => c.copy(epochStrategy = x))
    opt[Double]("epochRatio")
      .text("epoch delete ratio")
      .action((x,c) => c.copy(epochRatio = x))
    opt[Double]("epochDrop")
      .text("epoch delete drop decay")
      .action((x,c) => c.copy(epochDrop= x))

    opt[Int]("notIteration")
      .text("do not del anythig in fist  ith epoch")
      .action((x,c) => c.copy(notIteration = x))
    opt[String]( "optMethod")
      .text("optimization metho default is sgd")
      .action((x, c) => c.copy(optMethod = x))
    opt[Boolean]("useComp")
      .text("if use compressed data default if false")
      .action((x,c) => c.copy(useComp = x))
    opt[Boolean]("onlyComp")
      .text("if use only compressed data default if false")
      .action((x,c) => c.copy(onlyComp = x))
    opt[Boolean]("getGradient")
      .text("get gradients")
      .action((x,c) => c.copy(getGradient = x))

    opt[Boolean]("storeInitModel")
      .text("if save store init model")
      .action((x,c) => c.copy(storeInitModel = x))
    opt[String]("storeInitModelPath")
      .text("save init model in ...")
      .action((x,c) => c.copy(storeInitModelPath = Some(x)))
    opt[String]("storeInitStatePath")
      .text("save init state in ...")
      .action((x,c) => c.copy(storeInitStatePath = Some(x)))
    opt[Boolean]("storeTrainedModel")
      .text("if save trained model")
      .action((x,c) => c.copy(storeTrainedModel = x))
    opt[String]("storeTrainedModelPath")
      .text("save trained model in ...")
      .action((x,c) => c.copy(storeTrainedModelPath = Some(x)))
    opt[String]("storeTrainedStatePath")
      .text("storeTrainedStatePath")
      .action((x,c) => c.copy(storeTrainedStatePath = Some(x)))

    opt[Boolean]("zScore")
      .text("if z-score features")
      .action((x,c) => c.copy(zScore = x))
    opt[String]("lrScheduler")
      .text("sgd learningRate scheduler")
      .action((x,c) => c.copy(lrScheduler = Some(x)))
    opt[Boolean]("hasDropout")
      .text("wheather has droput")
      .action((x,c) => c.copy(hasDropout = x))
    opt[Int]("notEpoch")
      .text("not del epoch")
      .action((x,c) => c.copy(notEpoch = x))
    opt[Boolean]("hasbn")
      .text("has BN")
      .action((x,c) => c.copy(hasbn = x))
  }

  case class TestParams(
                         folder: String = "./",
                         model: String = "",
                         batchSize: Int = 128
                       )

  val testParser = new OptionParser[TestParams]("BigDL LeNet Test Example") {
    opt[String]('f', "folder")
      .text("the location of dataset")
      .action((x, c) => c.copy(folder = x))

    opt[String]('m', "model")
      .text("the location of model snapshot")
      .action((x, c) => c.copy(model = x))
      .required()
      .required()
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
  }

}
