package slpart.loaderTest

import java.text.SimpleDateFormat
import java.util.Date

import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter}
import org.apache.spark.SparkContext
import slpart.datatreating.Mnist

object MnistCompTest {
  import slpart.models.alexnet.Utils._

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
      val trainSamples = Mnist.trainSamples(param.folder,sc,param.classes,param.zScore,param.useComp,
        param.itqbitN,param.itqitN,param.itqratioN,param.upBound,param.splitN,param.isSparse)
      val validationSamples = Mnist.validationSamples(param.folder,sc,param.zScore)
      System.out.println(s"generate Aggregate Samples: ${(System.nanoTime()-start) * 1.0 / 1e9} seconds")

      sc.stop()
    })
  }

}
