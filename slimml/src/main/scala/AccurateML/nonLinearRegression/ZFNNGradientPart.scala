package AccurateML.nonLinearRegression

import java.io.File

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import redis.clients.jedis.Jedis
import AccurateML.blas.ZFUtils

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
  * Created by zhangfan on 16/11/17.
  */
class ZFNNGradientPart(fitmodel: NonlinearModel, xydata: RDD[LabeledPoint], r: Double,redisHost:String) extends Serializable {
  var model: NonlinearModel = fitmodel
  var dim = fitmodel.getDim()
  var data: RDD[LabeledPoint] = xydata
  var m: Int = data.cache().count().toInt
  var n: Int = data.first().features.size
  val ratio: Double = r


  /**
    * Return the objective function dimensionality which is essentially the model's dimensionality
    */
  def getDim(): Int = {
    return this.dim
  }

  /**
    * This method is inherited by Breeze DiffFunction. Given an input vector of weights it returns the
    * objective function and the first order derivative.
    * It operates using treeAggregate action on the training pair data.
    * It is essentially the same implementation as the one used for the Stochastic Gradient Descent
    * Partial subderivative vectors are calculated in the map step
    * val per = fitModel.eval(w, feat)
    * val gper = fitModel.grad(w, feat)
    * and are aggregated by summation in the reduce part.
    */
  def calculate(weights: BDV[Double], itN: Int): (Double, BDV[Double], Array[Double], Int) = {
    assert(dim == weights.length)
    val bcW = data.context.broadcast(weights)

    val fitModel: NonlinearModel = model
    val n: Int = dim
    val bcDim = data.context.broadcast(dim)


    val mapData = data.mapPartitions(pit => {
      val jedis = new Jedis(redisHost)
      val nnMapT = System.currentTimeMillis()
      val ggfs = new ArrayBuffer[(Double, Double, Double, BDV[Double], Double)]()
      while (pit.hasNext) {
        val inc = pit.next()
        val label = inc.label
        val features = inc.features
        val feat: BDV[Double] = new BDV[Double](features.toArray)
        val w: BDV[Double] = new BDV[Double](bcW.value.toArray)
        val per = fitModel.eval(w, feat)
        val gper = fitModel.grad(w, feat)
        val f1 = 0.5 * Math.pow(label - per, 2)
        val g1 = 2.0 * (per - label) * gper

        val t: Double = g1 dot g1
        val gn = math.sqrt(t)
        //        ggfs += Tuple3(gn, g1, f1)
        ggfs += Tuple5(gper.toArray.map(math.abs).sum, math.abs(per - label), gn, g1, f1)
      }
      var ggf = ggfs.toArray
      val partN = ggf.length
      jedis.append("nnMapT", "," + (System.currentTimeMillis() - nnMapT))
      jedis.append("partN", "," + partN)
      jedis.close()

      ggf.sortWith(_._3 > _._3).slice(0, (partN * ratio).toInt).map(t => Tuple3(t._3, t._4, t._5)).toIterator

      //      ggf.sortBy(t => (t._1 <= 1E-2, -math.abs(t._2))).slice(0,(partN*ratio).toInt).map(t=>Tuple3(t._3,t._4,t._5)).toIterator
      //      Sorting.quickSort(ggf)(Ordering.by[(Double, BDV[Double], Double), Double](-_._1))
      //      ggf.slice(0, (partN * ratio).toInt).toIterator
    })
    //    val allN = mapData.count()
    val (allGradN, allGrad, allF, allN) = mapData.treeAggregate((new ArrayBuffer[Double], BDV.zeros[Double](n), 0.0, 0))(
      seqOp = (c, v) => (c, v) match {
        case ((gradn, grad, f, n), (agn, ag, af)) =>
          gradn += agn
          (gradn, grad + ag, f + af, n + 1)
      },
      combOp = (c1, c2) => (c1, c2) match {
        case ((gradn1, grad1, f1, n1), (gradn2, grad2, f2, n2)) =>
          val gg = gradn1 ++ gradn2
          grad1 += grad2
          (gg, grad1, f1 + f2, n1 + n2)
      })

    return (allF, allGrad, allGradN.toArray, allN)
  }


}

object ZFNNGradientPart {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("test nonlinear")
    val sc = new SparkContext(conf)

    val initW: Double = args(0).toDouble
    val stepSize: Double = args(1).toDouble
    val numFeature: Int = args(2).toInt //10
    val hiddenNodesN: Int = args(3).toInt //5
    val itN: Int = args(4).toInt
    val testPath: String = args(5)
    val dataPath: String = args(6)
    val test100: Array[Double] = args(7).split(",").map(_.toDouble)

    val weightsPath = args(8)
    val isSparse = args(9).toBoolean
    val minPartN = args(10).toInt

    val redisHost = "172.18.11.97"

    //    val ratioL = if (test100) List(100) else List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100)
    val ratioL = test100

    val vecs = new ArrayBuffer[Vector]()
    val mesb = new ArrayBuffer[Double]()
    val nntimesb = new ArrayBuffer[Double]()

    for (r <- ratioL) {
      val dataTxt = sc.textFile(dataPath, minPartN) // "/Users/zhangfan/Documents/nonlinear.f10.n100.h5.txt"
      val dim = new NeuralNetworkModel(numFeature, hiddenNodesN).getDim()
      val w0 = if (initW == -1) {
        val iter = Source.fromFile(new File(weightsPath)).getLines()
        val weights = iter.next().split(",").map(_.toDouble)
        new BDV(weights)
      } else BDV(Array.fill(dim)(initW))

      val jedis = new Jedis(redisHost)
      jedis.flushAll()
      val ratio = r / 100.0
      val data = if (!isSparse) {
        dataTxt.map(line => {
          val vs = line.split(",").map(_.toDouble)
          val features = vs.slice(0, vs.size - 1)
          LabeledPoint(vs.last, Vectors.dense(features))
        })
      } else {
        MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPartN)
      }

      val splits = data.randomSplit(Array(0.8, 0.2), seed = System.currentTimeMillis())
      val train = if (testPath.size > 3) data.cache() else splits(0).cache()
      val test = if (testPath.size > 3) {
        println("testPath,\t" + testPath)
        //        MLUtils.loadLibSVMFile(sc, testPath)
        if (!isSparse) {
          sc.textFile(testPath).map(line => {
            val vs = line.split(",").map(_.toDouble)
            val features = vs.slice(0, vs.size - 1)
            LabeledPoint(vs.last, Vectors.dense(features))
          })
        } else {
          MLUtils.loadLibSVMFile(sc, testPath, numFeature, minPartN)
        }
      } else splits(1)

      //      val train = data.cache()
      var trainN = 0.0

      val model: NonlinearModel = new NeuralNetworkModel(numFeature, hiddenNodesN)
      val modelTrain: ZFNNGradientPart = new ZFNNGradientPart(model, train, ratio,redisHost)
      val hissb = new StringBuilder()
      val w = w0.copy
      for (i <- 1 to itN) {
        val (f1, g1, gn, itTrainN) = modelTrain.calculate(w, i)
        hissb.append("," + f1 / itTrainN)
        val itStepSize = stepSize / itTrainN / math.sqrt(i) //this is stepSize for each iteration
        w -= itStepSize * g1
        trainN += itTrainN

      }
      trainN /= itN
      vecs += Vectors.dense(w.toArray)

      //      val test = if(!isSparse){
      //        sc.textFile(testPath).map(line => {
      //          val vs = line.split(",").map(_.toDouble)
      //          val features = vs.slice(0, vs.size - 1)
      //          LabeledPoint(vs.last, Vectors.dense(features))
      //        })
      //      } else{
      //        MLUtils.loadLibSVMFile(sc, testPath, numFeature, minPartN)
      //      }
      val MSE = test.map { point =>
        val prediction = model.eval(w, new BDV[Double](point.features.toArray))
        (point.label, prediction)
      }.map { case (v, p) => 0.5 * math.pow((v - p), 2) }.mean()

      val partN = jedis.get("partN").split(",")
      val nnMapT = jedis.get("nnMapT").split(",").map(ZFUtils.zfParseDouble).filter(_ != None).map(_.get)
      jedis.close()
      println()
      println("dataPart," + data.getNumPartitions + ",trainPart," + train.getNumPartitions + ",testPart," + test.getNumPartitions)
      println("partN," + partN.slice(0, math.min(w0.length, 50)).mkString(","))
      println(",ratio," + ratio + ",itN," + itN + ",initW," + w0(0) + ",step," + stepSize + ",hiddenN," + hiddenNodesN + ",trainN," + trainN / 10000.0 + ",testN," + test.count() / 10000.0 + ",numFeatures," + numFeature)
      System.out.println("w0= ," + w0.slice(0, math.min(w0.length, 10)).toArray.mkString(","))
      System.out.println("w = ," + w.slice(0, math.min(w.length, 10)).toArray.mkString(","))
      System.out.println(",MSE, " + MSE + ",[" + hissb)
      println("nnMapT," + nnMapT.sum)

      mesb += MSE
      nntimesb += nnMapT.sum

    }
    val n = vecs.length

    println()
    println(this.getClass.getName + ",step," + stepSize + ",data," + dataPath)
    println("ratio,MSE,nnMapT")
    for (i <- vecs.toArray.indices) {
      println(ratioL(i) / 100.0 + "," + mesb(i) + "," + nntimesb(i))
    }


  }


}
